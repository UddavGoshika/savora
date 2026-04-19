const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const axios = require('axios');
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json({ limit: '20mb' }));
app.use(passport.initialize());

const PORT = process.env.PORT || 3000;
const GEMINI_KEY = process.env.GEMINI_KEY || "";
const OR_KEY = process.env.OR_KEY || "";
const GROQ_KEY = process.env.GROQ_KEY || "";
const MONGO_URI = process.env.MONGO_URI || "";
const JWT_SECRET = process.env.JWT_SECRET || "savora_secret_key_123";
const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID || "";
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || "";

let isDbConnected = false;

const recipeCache = new Map();
const CACHE_MAX_SIZE = 2000;

const connectDB = async () => {
    try {
        if (MONGO_URI && MONGO_URI.startsWith('mongodb')) {
            await mongoose.connect(MONGO_URI);
            isDbConnected = true;
            console.log('✓ MongoDB Atlas Connected');
        }
    } catch (err) {
        console.log('⚠ DB not connected:', err.message);
    }
};
connectDB();

const userRateLimits = new Map();
const rateLimitWindow = 5 * 60 * 60 * 1000;
const VISION_DAILY_LIMIT = 10;
const CHAT_DAILY_LIMIT = 50;

function checkRateLimit(userId, type) {
    if (!userId) return { allowed: true, remaining: 999 };
    const key = `${userId}_${type}`;
    const now = Date.now();
    const limit = type === 'vision' ? VISION_DAILY_LIMIT : CHAT_DAILY_LIMIT;
    
    if (!userRateLimits.has(key)) {
        userRateLimits.set(key, { count: 1, resetAt: now + rateLimitWindow });
        return { allowed: true, remaining: limit - 1 };
    }
    
    const limitData = userRateLimits.get(key);
    if (now > limitData.resetAt) {
        userRateLimits.set(key, { count: 1, resetAt: now + rateLimitWindow });
        return { allowed: true, remaining: limit - 1 };
    }
    
    if (limitData.count >= limit) {
        return { allowed: false, remaining: 0, resetAt: limitData.resetAt };
    }
    
    limitData.count++;
    return { allowed: true, remaining: limit - limitData.count };
}

function getFromMemoryCache(key) {
    if (recipeCache.has(key)) {
        const cached = recipeCache.get(key);
        cached.hits = (cached.hits || 0) + 1;
        return cached.data;
    }
    return null;
}

function setToMemoryCache(key, data) {
    if (recipeCache.size >= CACHE_MAX_SIZE) {
        const firstKey = recipeCache.keys().next().value;
        recipeCache.delete(firstKey);
    }
    recipeCache.set(key, { data, created: Date.now(), hits: 1 });
}

async function getFromDBCache(key) {
    if (!isDbConnected) return null;
    try {
        const cached = await RecipeModel.findOne({ ingredientsKey: key });
        if (cached) {
            setToMemoryCache(key, cached.recipeData);
            console.log('--- DB CACHE HIT ---');
            return cached.recipeData;
        }
    } catch (e) { console.log('DB error:', e.message); }
    return null;
}

async function saveToDBCache(key, data) {
    if (!isDbConnected) return;
    try {
        await RecipeModel.findOneAndUpdate(
            { ingredientsKey: key },
            { ingredientsKey: key, recipeData: data, createdAt: Date.now() },
            { upsert: true, new: true }
        );
    } catch (e) { console.log('DB save error:', e.message); }
}

async function generateWithAI(prompt) {
    if (OR_KEY) {
        try {
            const orResp = await axios.post("https://openrouter.ai/api/v1/chat/completions", {
                model: "meta-llama/llama-3.1-8b-instruct:free",
                messages: [{ role: "user", content: prompt }],
                max_tokens: 512,
                temperature: 0.7
            }, {
                headers: { 
                    "Authorization": `Bearer ${OR_KEY}`, 
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://savora-mldt.onrender.com",
                    "X-Title": "Savora"
                },
                timeout: 15000
            });
            return orResp.data.choices[0].message.content;
        } catch (orErr) { console.log('OpenRouter failed'); }
    }
    
    if (GROQ_KEY) {
        try {
            const groqResp = await axios.post("https://api.groq.com/openai/v1/chat/completions", {
                model: "llama-3.1-8b-instant",
                messages: [{ role: "user", content: prompt }],
                max_tokens: 512,
                temperature: 0.7
            }, {
                headers: { "Authorization": `Bearer ${GROQ_KEY}`, "Content-Type": "application/json" },
                timeout: 10000
            });
            return groqResp.data.choices[0].message.content;
        } catch (groqErr) { console.log('GroqCloud failed'); }
    }
    
    if (GEMINI_KEY) {
        try {
            const genAI = new GoogleGenerativeAI(GEMINI_KEY);
            const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
            const result = await model.generateContent(prompt);
            return (await result.response).text();
        } catch (gemErr) { console.log('Gemini failed'); }
    }
    
    return null;
}

async function detectIngredientsWithAI(base64Image) {
    if (!GEMINI_KEY) return ["Tomato", "Onion", "Garlic"];
    try {
        const genAI = new GoogleGenerativeAI(GEMINI_KEY);
        const visionModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
        const result = await visionModel.generateContent({
            contents: [{ role: 'user', parts: [
                { text: "List food ingredients. Comma separated." },
                { inlineData: { mimeType: "image/jpeg", data: base64Image } }
            ]}]
        });
        const text = (await result.response).text();
        return text.split(/[,,\n]/).map(i => i.trim()).filter(Boolean);
    } catch (e) { return ["Tomato", "Onion", "Garlic"]; }
}

const UserSchema = new mongoose.Schema({
    name: String,
    email: { type: String, unique: true, required: true },
    password: { type: String },
    googleId: String,
    avatar: String,
    plan: { type: String, default: 'FREE' },
    createdAt: { type: Date, default: Date.now }
});
const User = mongoose.model('User', UserSchema);

const RecipeSchema = new mongoose.Schema({
    ingredientsKey: { type: String, unique: true, index: true },
    recipeData: Object,
    createdAt: { type: Date, default: Date.now }
});
const RecipeModel = mongoose.model('Recipe', RecipeSchema);

const TrendingRecipeSchema = new mongoose.Schema({
    name: String,
    cuisine: String,
    difficulty: String,
    time_minutes: Number,
    calories: Number,
    ingredients: [String],
    steps: String,
    protein_g: Number,
    carbs_g: Number,
    fat_g: Number,
    imageUrl: String,
    createdAt: { type: Date, default: Date.now }
});
const TrendingModel = mongoose.model('TrendingRecipe', TrendingRecipeSchema);

if (GOOGLE_CLIENT_ID && GOOGLE_CLIENT_SECRET) {
    passport.use(new GoogleStrategy({
        clientID: GOOGLE_CLIENT_ID,
        clientSecret: GOOGLE_CLIENT_SECRET,
        callbackURL: process.env.GOOGLE_REDIRECT_URI
    }, async (accessToken, refreshToken, profile, done) => {
        try {
            let user = await User.findOne({ googleId: profile.id });
            if (!user) {
                user = await User.findOne({ email: profile.emails[0].value });
                if (user) {
                    user.googleId = profile.id;
                    user.avatar = profile.photos[0]?.value;
                    await user.save();
                } else {
                    user = new User({
                        name: profile.displayName,
                        email: profile.emails[0].value,
                        googleId: profile.id,
                        avatar: profile.photos[0]?.value
                    });
                    await user.save();
                }
            }
            return done(null, user);
        } catch (err) { return done(err, null); }
    }));
    passport.serializeUser((user, done) => done(null, user.id));
    passport.deserializeUser(async (id, done) => {
        try {
            const user = await User.findById(id);
            done(null, user);
        } catch (err) { done(err, null); }
    });
}

const trendingRecipes = [
    { name: "Butter Chicken", cuisine: "Indian", difficulty: "Medium", time_minutes: 45, calories: 450, ingredients: ["Chicken", "Butter", "Cream", "Tomatoes", "Garlic", "Ginger", "Garam Masala"], steps: "1. Marinate chicken in yogurt.\n2. Cook tomatoes with spices.\n3. Add chicken and simmer.\n4. Finish with cream and butter.", protein_g: 35, carbs_g: 15, fat_g: 28 },
    { name: "Palak Paneer", cuisine: "Indian", difficulty: "Easy", time_minutes: 30, calories: 320, ingredients: ["Paneer", "Spinach", "Onion", "Garlic", "Green Chili"], steps: "1. Blanch spinach.\n2. Make paste with onion.\n3. Add paneer cubes.\n4. Cook until tender.", protein_g: 18, carbs_g: 12, fat_g: 24 },
    { name: "Chicken Biryani", cuisine: "Indian", difficulty: "Hard", time_minutes: 60, calories: 520, ingredients: ["Basmati Rice", "Chicken", "Yogurt", "Onions", "Spices", "Saffron"], steps: "1. Marinate chicken.\n2. Fry onions.\n3. Layer rice and chicken.\n4. Dum cook on low flame.", protein_g: 28, carbs_g: 65, fat_g: 18 },
    { name: "Dal Tadka", cuisine: "Indian", difficulty: "Easy", time_minutes: 25, calories: 180, ingredients: ["Toor Dal", "Onion", "Tomato", "Garlic", "Cumin", "Ghee"], steps: "1. Cook dal until soft.\n2. Make tadka with spices.\n3. Mix together.\n4. Serve with rice.", protein_g: 12, carbs_g: 28, fat_g: 4 },
    { name: "Paneer Tikka", cuisine: "Indian", difficulty: "Medium", time_minutes: 35, calories: 290, ingredients: ["Paneer", "Yogurt", "Bell Pepper", "Onion", "Tikka Masala"], steps: "1. Marinate paneer in yogurt.\n2. Thread on skewers.\n3. Grill until charred.\n4. Serve with mint chutney.", protein_g: 16, carbs_g: 10, fat_g: 22 },
    { name: "Chole Bhature", cuisine: "Indian", difficulty: "Hard", time_minutes: 50, calories: 480, ingredients: ["Chickpeas", "Flour", "Onion", "Tomato", "Chole Masala"], steps: "1. Cook chickpeas.\n2. Make masala gravy.\n3. Prepare bhature dough.\n4. Fry and serve together.", protein_g: 15, carbs_g: 58, fat_g: 22 },
    { name: "Rajma Chawal", cuisine: "Indian", difficulty: "Medium", time_minutes: 45, calories: 420, ingredients: ["Kidney Beans", "Rice", "Onion", "Tomato", "Spices"], steps: "1. Soak and cook rajma.\n2. Make gravy.\n3. Cook rice.\n4. Serve with pickle.", protein_g: 16, carbs_g: 55, fat_g: 14 },
    { name: "Samosa", cuisine: "Indian", difficulty: "Medium", time_minutes: 40, calories: 260, ingredients: ["Flour", "Potato", "Peas", "Green Chili", "Cumin"], steps: "1. Make dough.\n2. Prepare stuffing.\n3. Shape samosas.\n4. Deep fry until golden.", protein_g: 4, carbs_g: 32, fat_g: 14 },
    { name: "Pav Bhaji", cuisine: "Indian", difficulty: "Medium", time_minutes: 40, calories: 380, ingredients: ["Potato", "Tomato", "Onion", "Pav Bread", "Butter", "Spices"], steps: "1. Cook vegetables.\n2. Mash and spiced.\n3. Toast pav with butter.\n4. Assemble and serve.", protein_g: 8, carbs_g: 45, fat_g: 18 },
    { name: "Idli Sambar", cuisine: "South Indian", difficulty: "Easy", time_minutes: 30, calories: 220, ingredients: ["Idli Rice", "Urad Dal", "Toor Dal", "Tomato", "Curry Leaves"], steps: "1. Ferment batter.\n2. Steam idlis.\n3. Make sambar.\n4. Serve hot.", protein_g: 8, carbs_g: 38, fat_g: 4 },
    { name: "Dosa", cuisine: "South Indian", difficulty: "Medium", time_minutes: 25, calories: 180, ingredients: ["Rice", "Urad Dal", "Methi Seeds", "Oil"], steps: "1. Soak and grind batter.\n2. Spread thin on tawa.\n3. Cook until crisp.\n4. Serve with sambar.", protein_g: 4, carbs_g: 30, fat_g: 6 },
    { name: "Vada Pav", cuisine: "Maharashtrian", difficulty: "Medium", time_minutes: 35, calories: 320, ingredients: ["Potato", "Chickpea Flour", "Garlic", "Chutney", "Pav Bread"], steps: "1. Make potato filling.\n2. Coat in gram flour.\n3. Fry until crisp.\n4. Assemble with chutney.", protein_g: 6, carbs_g: 40, fat_g: 16 },
    { name: "Poha", cuisine: "Maharashtrian", difficulty: "Easy", time_minutes: 20, calories: 250, ingredients: ["Flattened Rice", "Onion", "Peanuts", "Curry Leaves", "Mustard"], steps: "1. Rinse poha.\n2. Make tempering.\n3. Mix together.\n4. Top with sev.", protein_g: 5, carbs_g: 35, fat_g: 10 },
    { name: "Misal Pav", cuisine: "Maharashtrian", difficulty: "Medium", time_minutes: 35, calories: 340, ingredients: ["Sprouted Moth Beans", "Potato", "Onion", "Pav", "Farsan"], steps: "1. Cook sprouted beans.\n2. Make gravy with spices.\n3. Top with farsan.\n4. Serve with pav.", protein_g: 12, carbs_g: 38, fat_g: 16 },
    { name: "Bhelpuri", cuisine: "Indian", difficulty: "Easy", time_minutes: 15, calories: 200, ingredients: ["Puffed Rice", "Onion", "Tomato", "Peanuts", "Chutney"], steps: "1. Mix dry ingredients.\n2. Add chutney.\n3. Toss well.\n4. Serve immediately.", protein_g: 4, carbs_g: 30, fat_g: 8 },
    { name: "Chicken 65", cuisine: "South Indian", difficulty: "Medium", time_minutes: 30, calories: 340, ingredients: ["Chicken", "Curry Leaves", "Ginger", "Garlic", "Chili"], steps: "1. Marinate chicken.\n2. Fry until crispy.\n3. Temper with spices.\n4. Serve hot.", protein_g: 28, carbs_g: 8, fat_g: 22 },
    { name: "Fish Fry", cuisine: "Coastal Indian", difficulty: "Easy", time_minutes: 20, calories: 280, ingredients: ["Fish", "Turmeric", "Chili Powder", "Lemon"], steps: "1. Marinate fish.\n2. Coat with spices.\n3. Shallow fry.\n4. Serve with onion.", protein_g: 24, carbs_g: 6, fat_g: 18 },
    { name: "Prawns Curry", cuisine: "Coastal Indian", difficulty: "Medium", time_minutes: 30, calories: 260, ingredients: ["Prawns", "Coconut Milk", "Curry Leaves", "Spices"], steps: "1. Clean prawns.\n2. Make curry base.\n3. Add prawns.\n4. Simmer until done.", protein_g: 22, carbs_g: 10, fat_g: 14 },
    { name: "Egg Curry", cuisine: "Indian", difficulty: "Easy", time_minutes: 25, calories: 280, ingredients: ["Eggs", "Onion", "Tomato", "Spices", "Coriander"], steps: "1. Boil eggs.\n2. Make gravy.\n3. Add eggs.\n4. Cook for 5 mins.", protein_g: 14, carbs_g: 12, fat_g: 20 },
    { name: "Mutton Curry", cuisine: "Indian", difficulty: "Hard", time_minutes: 60, calories: 380, ingredients: ["Mutton", "Onion", "Tomato", "Ginger", "Garlic", "Garam Masala"], steps: "1. Marinate mutton.\n2. Cook onions.\n3. Add spices and meat.\n4. Dum cook for 1 hour.", protein_g: 32, carbs_g: 8, fat_g: 26 },
    { name: "Kadhi Pakora", cuisine: "Indian", difficulty: "Medium", time_minutes: 35, calories: 220, ingredients: ["Yogurt", "Chickpea Flour", "Onion", "Curry Leaves"], steps: "1. Make pakoras.\n2. Prepare kadhi.\n3. Combine both.\n4. Serve with rice.", protein_g: 8, carbs_g: 24, fat_g: 12 },
    { name: "Gujarati Dal", cuisine: "Gujarati", difficulty: "Easy", time_minutes: 25, calories: 180, ingredients: ["Tuvar Dal", "Jaggery", "Tamarind", "Peanuts"], steps: "1. Cook dal.\n2. Add tadka.\n3. Sweeten with jaggery.\n4. Serve with roti.", protein_g: 10, carbs_g: 28, fat_g: 4 },
    { name: "Thepla", cuisine: "Gujarati", difficulty: "Easy", time_minutes: 25, calories: 200, ingredients: ["Wheat Flour", "Methi", "Curd", "Turmeric"], steps: "1. Make dough.\n2. Add methi.\n3. Roll and cook.\n4. Serve with pickle.", protein_g: 6, carbs_g: 32, fat_g: 6 },
    { name: "Khandvi", cuisine: "Gujarati", difficulty: "Medium", time_minutes: 30, calories: 160, ingredients: ["Besan", "Yogurt", "Turmeric", "Oil"], steps: "1. Cook batter.\n2. Spread thin.\n3. Roll with filling.\n4. Temper with spices.", protein_g: 6, carbs_g: 20, fat_g: 8 },
    { name: "Dhokla", cuisine: "Gujarati", difficulty: "Easy", time_minutes: 30, calories: 150, ingredients: ["Besan", "Semolina", "Yogurt", "Eno"], steps: "1. Ferment batter.\n2. Steam in mold.\n3. Temper with mustard.\n4. Serve with chutney.", protein_g: 5, carbs_g: 24, fat_g: 4 },
    { name: "Aloo Gobi", cuisine: "North Indian", difficulty: "Easy", time_minutes: 25, calories: 180, ingredients: ["Potato", "Cauliflower", "Onion", "Tomato", "Spices"], steps: "1. Chop vegetables.\n2. Temper with spices.\n3. Add vegetables.\n4. Stir fry until tender.", protein_g: 4, carbs_g: 22, fat_g: 10 },
    { name: "Aloo Mutter", cuisine: "North Indian", difficulty: "Easy", time_minutes: 25, calories: 190, ingredients: ["Potato", "Green Peas", "Onion", "Tomato", "Spices"], steps: "1. Cook potatoes.\n2. Make gravy.\n3. Add peas.\n4. Serve with roti.", protein_g: 5, carbs_g: 26, fat_g: 8 },
    { name: "Baingan Bharta", cuisine: "North Indian", difficulty: "Medium", time_minutes: 30, calories: 160, ingredients: ["Brinjal", "Onion", "Tomato", "Garlic", "Green Chili"], steps: "1. Roast brinjal.\n2. Make masala.\n3. Mash and mix.\n4. Serve with roti.", protein_g: 3, carbs_g: 14, fat_g: 10 },
    { name: "Gobhi Musallam", cuisine: "North Indian", difficulty: "Medium", time_minutes: 35, calories: 200, ingredients: ["Cauliflower", "Cashew", "Cream", "Spices"], steps: "1. Clean cauliflower.\n2. Make sauce.\n3. Cook until tender.\n4. Garnish with cream.", protein_g: 6, carbs_g: 18, fat_g: 12 },
    { name: "Mattar Paneer", cuisine: "North Indian", difficulty: "Easy", time_minutes: 25, calories: 280, ingredients: ["Paneer", "Green Peas", "Onion", "Tomato", "Cream"], steps: "1. Make gravy.\n2. Add paneer.\n3. Add peas.\n4. Finish with cream.", protein_g: 14, carbs_g: 16, fat_g: 20 },
    { name: "Shahi Paneer", cuisine: "North Indian", difficulty: "Medium", time_minutes: 30, calories: 320, ingredients: ["Paneer", "Cashew", "Cream", "Kewra", "Spices"], steps: "1. Make royal gravy.\n2. Add paneer.\n3. Simmer.\n4. Serve with naan.", protein_g: 14, carbs_g: 14, fat_g: 24 },
    { name: "Kadhai Paneer", cuisine: "North Indian", difficulty: "Medium", time_minutes: 30, calories: 300, ingredients: ["Paneer", "Green Peppers", "Onion", "Tomato", "Kadhai Masala"], steps: "1. Stir fry peppers.\n2. Add paneer.\n3. Add kadhai masala.\n4. Serve hot.", protein_g: 14, carbs_g: 12, fat_g: 22 },
    { name: "Achari Paneer", cuisine: "North Indian", difficulty: "Easy", time_minutes: 25, calories: 290, ingredients: ["Paneer", "Pickle Masala", "Onion", "Tomato"], steps: "1. Make gravy.\n2. Add pickle masala.\n3. Add paneer.\n4. Serve with naan.", protein_g: 14, carbs_g: 10, fat_g: 22 },
    { name: "Vegetable Biryani", cuisine: "Indian", difficulty: "Hard", time_minutes: 55, calories: 360, ingredients: ["Basmati Rice", "Mixed Vegetables", "Yogurt", "Onion", "Biryani Masala"], steps: "1. Cook rice partially.\n2. Make vegetable gravy.\n3. Layer with saffron.\n4. Dum cook.", protein_g: 8, carbs_g: 52, fat_g: 14 },
    { name: "Hyderabadi Biryani", cuisine: "Hyderabadi", difficulty: "Hard", time_minutes: 75, calories: 420, ingredients: ["Mutton", "Basmati Rice", "Yogurt", "Fried Onions", "Raw Mango"], steps: "1. Marinate meat.\n2. Make rice.\n3. Layer properly.\n4. Dum for 1 hour.", protein_g: 24, carbs_g: 48, fat_g: 18 },
    { name: "Kashmiri Pulao", cuisine: "Kashmiri", difficulty: "Medium", time_minutes: 40, calories: 320, ingredients: ["Basmati Rice", "Almonds", "Saffron", "Milk", "Ghee"], steps: "1. Cook rice.\n2. Add nuts.\n3. Saffron milk.\n4. Serve with curry.", protein_g: 8, carbs_g: 42, fat_g: 14 },
    { name: "Jeera Rice", cuisine: "Indian", difficulty: "Easy", time_minutes: 20, calories: 200, ingredients: ["Basmati Rice", "Cumin", "Ghee", "Lemon"], steps: "1. Toast cumin.\n2. Add rice.\n3. Cook with water.\n4. Fluff and serve.", protein_g: 4, carbs_g: 30, fat_g: 6 },
    { name: "Vegetable Pulao", cuisine: "Indian", difficulty: "Medium", time_minutes: 30, calories: 280, ingredients: ["Basmati Rice", "Carrot", "Beans", "Peas", "Onion", "Ghee"], steps: "1. Cook vegetables.\n2. Add rice.\n3. Add spices.\n4. Serve with raita.", protein_g: 6, carbs_g: 38, fat_g: 12 },
    { name: "Coconut Rice", cuisine: "South Indian", difficulty: "Easy", time_minutes: 20, calories: 260, ingredients: ["Rice", "Coconut", "Curry Leaves", "Peanuts", "Mustard"], steps: "1. Toast spices.\n2. Add coconut.\n3. Mix with rice.\n4. Serve with pickle.", protein_g: 4, carbs_g: 32, fat_g: 14 },
    { name: "Lemon Rice", cuisine: "South Indian", difficulty: "Easy", time_minutes: 18, calories: 220, ingredients: ["Rice", "Lemon", "Curry Leaves", "Peanuts", "Green Chili"], steps: "1. Make tempering.\n2. Add lemon juice.\n3. Mix with rice.\n4. Serve immediately.", protein_g: 4, carbs_g: 30, fat_g: 8 },
    { name: "Tomato Rice", cuisine: "South Indian", difficulty: "Easy", time_minutes: 20, calories: 230, ingredients: ["Rice", "Tomato", "Onion", "Curry Leaves", "Spices"], steps: "1. Cook tomatoes.\n2. Add rice.\n3. Mix spices.\n4. Serve with papad.", protein_g: 4, carbs_g: 32, fat_g: 8 },
    { name: "Curd Rice", cuisine: "South Indian", difficulty: "Easy", time_minutes: 15, calories: 180, ingredients: ["Rice", "Curd", "Curry Leaves", "Mustard", "Ginger"], steps: "1. Cook and cool rice.\n2. Mix with curd.\n3. Add tempering.\n4. Serve cold.", protein_g: 6, carbs_g: 24, fat_g: 6 },
    { name: "Rasam", cuisine: "South Indian", difficulty: "Easy", time_minutes: 20, calories: 80, ingredients: ["Tamarind", "Tomato", "Curry Leaves", "Pepper", "Cumin"], steps: "1. Extract tamarind water.\n2. Boil with spices.\n3. Add tempering.\n4. Serve hot.", protein_g: 2, carbs_g: 12, fat_g: 2 },
    { name: "Sambar", cuisine: "South Indian", difficulty: "Medium", time_minutes: 30, calories: 160, ingredients: ["Toor Dal", "Vegetables", "Tamarind", "Curry Leaves", "Spices"], steps: "1. Cook dal.\n2. Add vegetables.\n3. Make masala.\n4. Simmer together.", protein_g: 10, carbs_g: 22, fat_g: 4 },
    { name: "Mango Pickle", cuisine: "Indian", difficulty: "Easy", time_minutes: 30, calories: 60, ingredients: ["Raw Mango", "Mustard Oil", "Fenugreek", "Salt"], steps: "1. Cut mango.\n2. Apply salt.\n3. Add spices.\n4. Store in oil.", protein_g: 0, carbs_g: 8, fat_g: 4 },
    { name: "Lemon Pickle", cuisine: "Indian", difficulty: "Easy", time_minutes: 20, calories: 40, ingredients: ["Lemon", "Salt", "Chili", "Oil"], steps: "1. Slice lemons.\n2. Add salt.\n3. Add oil.\n4. Store for 2 days.", protein_g: 0, carbs_g: 6, fat_g: 2 },
    { name: "Mixed Vegetable Curry", cuisine: "Indian", difficulty: "Easy", time_minutes: 25, calories: 180, ingredients: ["Mixed Vegetables", "Onion", "Tomato", "Ginger", "Garlic"], steps: "1. Chop vegetables.\n2. Make gravy.\n3. Add vegetables.\n4. Cook until done.", protein_g: 4, carbs_g: 20, fat_g: 10 },
    { name: "Vegetable Kofta", cuisine: "North Indian", difficulty: "Medium", time_minutes: 45, calories: 280, ingredients: ["Vegetables", "Paneer", "Cashew", "Cream", "Spices"], steps: "1. Make koftas.\n2. Make gravy.\n3. Add koftas.\n4. Serve with naan.", protein_g: 10, carbs_g: 22, fat_g: 18 },
    { name: "Malai Kofta", cuisine: "North Indian", difficulty: "Medium", time_minutes: 45, calories: 340, ingredients: ["Paneer", "Khoya", "Cream", "Tomato", "Cashew"], steps: "1. Make koftas.\n2. Make creamy gravy.\n3. Add koftas.\n4. Garnish with cream.", protein_g: 12, carbs_g: 20, fat_g: 24 },
    { name: "Navratan Korma", cuisine: "Mughlai", difficulty: "Medium", time_minutes: 40, calories: 320, ingredients: ["Paneer", "Mixed Fruits", "Khoya", "Cream", "Nuts"], steps: "1. Make korma base.\n2. Add paneer.\n3. Add fruits.\n4. Serve with pulao.", protein_g: 12, carbs_g: 22, fat_g: 22 },
    { name: "Shahi Egg Curry", cuisine: "North Indian", difficulty: "Easy", time_minutes: 25, calories: 280, ingredients: ["Eggs", "Cream", "Cashew", "Kewra", "Spices"], steps: "1. Boil eggs.\n2. Make royal gravy.\n3. Add eggs.\n4. Serve with naan.", protein_g: 14, carbs_g: 10, fat_g: 20 }
];

async function seedTrendingRecipes() {
    if (!isDbConnected) return;
    try {
        const count = await TrendingModel.countDocuments();
        if (count === 0) {
            await TrendingModel.insertMany(trendingRecipes);
            console.log(`✓ Seeded ${trendingRecipes.length} trending recipes`);
        } else {
            console.log(`✓ ${count} trending recipes already exist`);
        }
    } catch (e) { console.log('Seed error:', e.message); }
}
setTimeout(seedTrendingRecipes, 2000);

app.post('/api/auth/register', async (req, res) => {
    try {
        const { name, email, password } = req.body;
        let user = await User.findOne({ email });
        if (user) return res.status(400).json({ error: "User already exists" });
        const hashedPassword = await bcrypt.hash(password, 10);
        user = new User({ name, email, password: hashedPassword });
        await user.save();
        const token = jwt.sign({ id: user._id }, JWT_SECRET);
        res.json({ token, user: { name: user.name, email: user.email, plan: user.plan } });
    } catch (err) { res.status(500).json({ error: "Registration failed" }); }
});

app.post('/api/auth/login', async (req, res) => {
    try {
        const { email, password } = req.body;
        const user = await User.findOne({ email });
        if (!user) return res.status(400).json({ error: "User not found" });
        if (user.password) {
            const isMatch = await bcrypt.compare(password, user.password);
            if (!isMatch) return res.status(400).json({ error: "Invalid credentials" });
        } else if (user.googleId) {
            return res.status(400).json({ error: "Use Google Sign-In" });
        }
        const token = jwt.sign({ id: user._id }, JWT_SECRET);
        res.json({ token, user: { name: user.name, email: user.email, plan: user.plan } });
    } catch (err) { res.status(500).json({ error: "Login failed" }); }
});

app.post('/api/auth/google', async (req, res) => {
    try {
        const { email, name, googleId, avatar } = req.body;
        let user = await User.findOne({ email });
        if (!user) {
            user = new User({ email, name, googleId, avatar });
            await user.save();
        } else if (!user.googleId) {
            user.googleId = googleId;
            user.avatar = avatar;
            await user.save();
        }
        const token = jwt.sign({ id: user._id }, JWT_SECRET);
        res.json({ token, user: { name: user.name, email: user.email, plan: user.plan } });
    } catch (err) { res.status(500).json({ error: "Google Auth failed" }); }
});

app.get('/api/auth/me', async (req, res) => {
    const authHeader = req.headers.authorization;
    if (!authHeader) return res.status(401).json({ error: "No token" });
    const token = authHeader.split(' ')[1];
    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        const user = await User.findById(decoded.id).select('-password');
        if (!user) return res.status(404).json({ error: "User not found" });
        res.json(user);
    } catch (e) { res.status(401).json({ error: "Invalid token" }); }
});

app.get('/api/trending', async (req, res) => {
    try {
        const recipes = await TrendingModel.find().limit(50);
        res.json({ recipes });
    } catch (err) { res.status(500).json({ error: "Failed to fetch trending" }); }
});

app.post('/api/generate-recipe', async (req, res) => {
    try {
        const { ingredients } = req.body;
        if (!ingredients?.length) return res.status(400).json({ error: "Missing ingredients" });

        const cacheKey = ingredients.map(i => i.toLowerCase().trim()).sort().join('|');

        let cachedRecipe = getFromMemoryCache(cacheKey);
        if (cachedRecipe) return res.json(cachedRecipe);

        cachedRecipe = await getFromDBCache(cacheKey);
        if (cachedRecipe) return res.json(cachedRecipe);

        console.log('--- CACHE MISS (Calling AI) ---');
        const prompt = `Create a healthy recipe using: ${ingredients.join(", ")}. JSON: {"name":"","cuisine":"","difficulty":"Easy","time_minutes":20,"calories":350,"ingredients":[""],"steps":"","protein_g":20,"carbs_g":30,"fat_g":10}`;
        const recipeContent = await generateWithAI(prompt);
        if (!recipeContent) return res.status(500).json({ error: "AI unavailable" });

        const recipeParsed = parseJsonSafe(recipeContent);
        if (recipeParsed) {
            setToMemoryCache(cacheKey, recipeParsed);
            await saveToDBCache(cacheKey, recipeParsed);
            console.log('✓ Recipe saved to DB:', recipeParsed.name);
        }
        res.json(recipeParsed || getDefaultRecipe(ingredients));
    } catch (err) { res.status(500).json({ error: "Recipe failed" }); }
});

function getDefaultRecipe(ingredients) {
    return { name: "Quick " + (ingredients[0] || "Veg") + " Dish", cuisine: "Fusion", difficulty: "Easy", time_minutes: 20, calories: 350, ingredients: ingredients || [], steps: "1. Prepare.\n2. Cook.\n3. Serve.", protein_g: 15, carbs_g: 30, fat_g: 10 };
}

app.post('/api/detect-ingredients', async (req, res) => {
    const authHeader = req.headers.authorization;
    let userId = null;
    if (authHeader) {
        try {
            const token = authHeader.split(' ')[1];
            const decoded = jwt.verify(token, JWT_SECRET);
            userId = decoded.id;
        } catch (e) {}
    }
    const rateCheck = checkRateLimit(userId, 'vision');
    if (!rateCheck.allowed) return res.status(429).json({ error: "Vision limit reached. Reset at: " + new Date(rateCheck.resetAt).toLocaleTimeString() });
    
    try {
        const { base64Image } = req.body;
        if (!base64Image) return res.status(400).json({ error: "Image required" });
        const ingredients = await detectIngredientsWithAI(base64Image);
        res.json({ ingredients, remaining: rateCheck.remaining });
    } catch (err) { res.status(500).json({ error: "Vision failed" }); }
});

app.post('/api/sous-chef', async (req, res) => {
    const authHeader = req.headers.authorization;
    let userId = null;
    if (authHeader) {
        try {
            const token = authHeader.split(' ')[1];
            const decoded = jwt.verify(token, JWT_SECRET);
            userId = decoded.id;
        } catch (e) {}
    }
    const rateCheck = checkRateLimit(userId, 'chat');
    if (!rateCheck.allowed) return res.status(429).json({ error: "Chat limit reached. Reset at: " + new Date(rateCheck.resetAt).toLocaleTimeString() });
    
    try {
        const { question } = req.body;
        if (!question) return res.status(400).json({ error: "Question required" });
        const prompt = `Cooking assistant. Brief: ${question}`;
        const answer = await generateWithAI(prompt);
        res.json({ answer: answer || "I'm here to help!", remaining: rateCheck.remaining });
    } catch (err) { res.status(500).json({ error: "Chef busy" }); }
});

app.post('/api/weekly-plan', async (req, res) => {
    try {
        const { members, diet, ingredients } = req.body;
        const prompt = `7-day meal plan JSON.`;
        const planContent = await generateWithAI(prompt);
        const plan = parseJsonSafe(planContent);
        res.json(plan || getDefaultWeeklyPlan());
    } catch (err) { res.status(500).json({ error: "Meal plan failed" }); }
});

function getDefaultWeeklyPlan() {
    return {
        "Monday": { "Breakfast": "Oatmeal", "Lunch": "Dal rice", "Dinner": "Veg stir-fry" },
        "Tuesday": { "Breakfast": "Paratha", "Lunch": "Rajma", "Dinner": "Grilled chicken" },
        "Wednesday": { "Breakfast": "Idli", "Lunch": "Paneer", "Dinner": "Fish curry" },
        "Thursday": { "Breakfast": "Poha", "Lunch": "Biryani", "Dinner": "Soup" },
        "Friday": { "Breakfast": "Toast", "Lunch": "Pulao", "Dinner": "Chicken" },
        "Saturday": { "Breakfast": "Fruit", "Lunch": "Mix veg", "Dinner": "Pasta" },
        "Sunday": { "Breakfast": "Pancakes", "Lunch": "Family", "Dinner": "Salad" }
    };
}

app.post('/api/decide', async (req, res) => {
    try {
        const { budget, diet } = req.body;
        const prompt = `3 dinner options under ₹${budget || 100}. JSON array.`;
        const optionsContent = await generateWithAI(prompt);
        const options = parseJsonSafe(optionsContent);
        res.json({ options: options || getDefaultDecisions() });
    } catch (err) { res.status(500).json({ error: "Decide failed" }); }
});

function getDefaultDecisions() {
    return [
        { name: "Dal Tadka", calories: 280, time: "15 min", difficulty: "Easy", cuisine: "Indian" },
        { name: "Chicken Stir-Fry", calories: 420, time: "25 min", difficulty: "Medium", cuisine: "Asian" },
        { name: "Veggie Pulao", calories: 350, time: "20 min", difficulty: "Easy", cuisine: "Indian" }
    ];
}

app.post('/api/family-plan', async (req, res) => {
    try {
        const { members, diseases, diet } = req.body;
        const prompt = `Weekly plan JSON.`;
        const planContent = await generateWithAI(prompt);
        const plan = parseJsonSafe(planContent);
        res.json(plan || getDefaultWeeklyPlan());
    } catch (err) { res.status(500).json({ error: "Family plan failed" }); }
});

function parseJsonSafe(text) {
    try {
        if (!text) return null;
        const start = text.indexOf("{");
        const end = text.lastIndexOf("}");
        if (start === -1 || end === -1) return null;
        return JSON.parse(text.substring(start, end + 1));
    } catch (e) { return null; }
}

app.get('/api/health', (req, res) => {
    res.json({ status: "ok", db: isDbConnected, cache: recipeCache.size });
});

app.listen(PORT, '0.0.0.0', () => console.log(`Savora AI server on port ${PORT}`));
