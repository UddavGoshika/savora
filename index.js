const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const axios = require('axios');
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json({ limit: '20mb' }));

// ══ CONFIG (Strictly from .env) ══════════════════════════════
const PORT = process.env.PORT || 3000;
const GEMINI_KEY = process.env.GEMINI_KEY || "";
const OR_KEY = process.env.OR_KEY || "";
const MONGO_URI = process.env.MONGO_URI || "";
const JWT_SECRET = process.env.JWT_SECRET || "savora_secret_key_123";

if (!GEMINI_KEY || !OR_KEY || !MONGO_URI) {
    console.error("FATAL ERROR: Environment variables missing. Create a .env file.");
}

// CONNECT TO DB (Retry mechanism)
mongoose.connect(MONGO_URI)
  .then(() => console.log('Connected to Ingredia DB'))
  .catch(err => console.error('DB Connection Failed:', err));

const genAI = new GoogleGenerativeAI(GEMINI_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

// ══ DATABASE SCHEMAS ═══════════════════════════════════════════
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
    ingredientsKey: { type: String, unique: true, index: true }, // Sorted string of ingredients
    recipeData: Object,
    createdAt: { type: Date, default: Date.now }
});
const RecipeModel = mongoose.model('Recipe', RecipeSchema);

// ══ AUTH ENDPOINTS ═════════════════════════════════════════════

// 1. Register
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
    } catch (err) {
        res.status(500).json({ error: "Registration failed" });
    }
});

// 2. Login
app.post('/api/auth/login', async (req, res) => {
    try {
        const { email, password } = req.body;
        const user = await User.findOne({ email });
        if (!user) return res.status(400).json({ error: "User not found" });

        if (user.password) {
            const isMatch = await bcrypt.compare(password, user.password);
            if (!isMatch) return res.status(400).json({ error: "Invalid credentials" });
        } else if (user.googleId) {
            return res.status(400).json({ error: "Please use Google Sign-In for this account" });
        }

        const token = jwt.sign({ id: user._id }, JWT_SECRET);
        res.json({ token, user: { name: user.name, email: user.email, plan: user.plan } });
    } catch (err) {
        res.status(500).json({ error: "Login failed" });
    }
});

// 3. Google Auth (Simple version for mobile integration)
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
    } catch (err) {
        res.status(500).json({ error: "Google Auth failed" });
    }
});

// ══ AI ENDPOINTS ═══════════════════════════════════════════════

app.post('/api/generate-recipe', async (req, res) => {
    try {
        const { ingredients } = req.body;
        if (!ingredients || !ingredients.length) return res.status(400).json({ error: "Missing ingredients" });

        const cacheKey = ingredients.map(i => i.toLowerCase().trim()).sort().join('|');

        const cachedRecipe = await RecipeModel.findOne({ ingredientsKey: cacheKey });
        if (cachedRecipe) {
            console.log('--- DB CACHE HIT ---');
            return res.json(cachedRecipe.recipeData);
        }

        console.log('--- DB CACHE MISS (Calling AI) ---');
        const prompt = `Create a single healthy recipe using: ${ingredients.join(", ")}. Respond ONLY with valid JSON (no markdown): {"name":"","cuisine":"","difficulty":"Easy","time_minutes":20,"calories":350,"ingredients":[],"steps":"","protein_g":0,"carbs_g":0,"fat_g":0}`;
        
        let recipeContent;
        try {
            const result = await model.generateContent(prompt);
            recipeContent = (await result.response).text();
        } catch (geminiErr) {
            const orResp = await axios.post("https://openrouter.ai/api/v1/chat/completions", {
                model: "meta-llama/llama-3.1-8b-instruct:free",
                messages: [{ role: "user", content: prompt }]
            }, {
                headers: { "Authorization": `Bearer ${OR_KEY}`, "Content-Type": "application/json" }
            });
            recipeContent = orResp.data.choices[0].message.content;
        }

        const recipeParsed = parseJsonSafe(recipeContent);
        
        if (recipeParsed) {
            await new RecipeModel({
                ingredientsKey: cacheKey,
                recipeData: recipeParsed
            }).save();
        }

        res.json(recipeParsed);
    } catch (err) {
        console.error("Generator Error:", err);
        res.status(500).json({ error: "Recipe generation failed" });
    }
});

app.post('/api/detect-ingredients', async (req, res) => {
    try {
        const { base64Image } = req.body;
        const parts = [
            { text: "List ONLY food ingredients. No bullets. If none, write NONE" },
            { inlineData: { mimeType: "image/jpeg", data: base64Image } }
        ];
        const result = await model.generateContent({ contents: [{ role: 'user', parts }] });
        res.json({ ingredients: (await result.response).text().trim().split('\n').filter(Boolean) });
    } catch (err) { res.status(500).json({ error: "Vision failed" }); }
});

app.post('/api/sous-chef', async (req, res) => {
    try {
        const { question } = req.body;
        const result = await model.generateContent(`You are Ingredia Chef. Answer briefly: ${question}`);
        res.json({ answer: (await result.response).text().trim() });
    } catch (err) { res.status(500).json({ error: "Chef is busy" }); }
});

function parseJsonSafe(text) {
    try {
        const start = text.indexOf("{");
        const end = text.lastIndexOf("}");
        if (start === -1 || end === -1) return null;
        return JSON.parse(text.substring(start, end + 1));
    } catch (e) { return null; }
}

app.listen(PORT, '0.0.0.0', () => console.log(`Ingredia AI caching server running on port ${PORT}`));
