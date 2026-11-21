import "dotenv/config";
import express, { type Request, type Response } from "express";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { GoogleGenAI, Type, type Schema } from "@google/genai";
import { createServer as createViteServer } from "vite";
import { generateDemoRephrase } from "./demoRephrase.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const MODEL_NAME = "gemini-4-hydrogen";

interface RephraseBody {
  text?: unknown;
  style?: unknown;
  tone?: unknown;
  length?: unknown;
  clarity?: unknown;
  fixGrammar?: unknown;
  preserveMeaning?: unknown;
  variationSeed?: unknown;
}

const VALID_STYLES = new Set([
  "Professional",
  "Casual",
  "Academic",
  "Creative",
  "Friendly",
]);

const VALID_TONES = new Set([
  "Neutral",
  "Confident",
  "Diplomatic",
  "Empathetic",
  "Persuasive",
]);

const VALID_LENGTHS = new Set(["Shorter", "Balanced", "Expanded"]);
const VALID_CLARITIES = new Set(["Standard", "Clear & Direct", "Simplified"]);

const REPHRASE_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    rewrittenText: {
      type: Type.STRING,
      description:
        "The complete rewritten version of the user's text adhering to the requested style, tone, length, clarity, and grammar settings while preserving the core meaning.",
    },
    meaningPreservationNote: {
      type: Type.STRING,
      description:
        "A concise 1-2 sentence explanation confirming how the original intent, facts, and core message were preserved.",
    },
    toneSummary: {
      type: Type.STRING,
      description:
        "A crisp 1-sentence summary of the stylistic, structural, and clarity shifts applied.",
    },
    grammarFixes: {
      type: Type.ARRAY,
      description:
        "Specific grammar, punctuation, agreement, or mechanical errors corrected from the original text. Return an empty array if the original text has no grammatical issues.",
      items: {
        type: Type.OBJECT,
        properties: {
          original: {
            type: Type.STRING,
            description: "The exact phrase or clause from the original text.",
          },
          corrected: {
            type: Type.STRING,
            description: "The grammatically corrected form.",
          },
          issueType: {
            type: Type.STRING,
            description:
              "Short category label such as Grammar, Punctuation, Agreement, Syntax, or Conciseness.",
          },
          explanation: {
            type: Type.STRING,
            description: "Brief explanation of why the correction improves the writing.",
          },
        },
        required: ["original", "corrected", "issueType", "explanation"],
      },
    },
    wordingSuggestions: {
      type: Type.ARRAY,
      description:
        "3 to 5 high-impact vocabulary or phrasing upgrades. Each targetPhrase MUST be an exact substring present in rewrittenText so the user can swap it directly if desired.",
      items: {
        type: Type.OBJECT,
        properties: {
          targetPhrase: {
            type: Type.STRING,
            description:
              "An exact phrase that appears verbatim inside rewrittenText.",
          },
          suggestedAlternative: {
            type: Type.STRING,
            description:
              "An alternative stronger, sharper, or more nuanced phrasing that seamlessly replaces targetPhrase.",
          },
          reason: {
            type: Type.STRING,
            description: "Brief rationale explaining the nuance or benefit of this alternative.",
          },
        },
        required: ["targetPhrase", "suggestedAlternative", "reason"],
      },
    },
  },
  required: [
    "rewrittenText",
    "meaningPreservationNote",
    "toneSummary",
    "grammarFixes",
    "wordingSuggestions",
  ],
};

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === "MY_GEMINI_API_KEY") {
    throw new Error(
      "GEMINI_API_KEY is not configured. Please check the Settings > Secrets panel."
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function isTransientError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err);
  return (
    message.includes("429") ||
    message.includes("503") ||
    message.includes("500") ||
    message.includes("UNAVAILABLE") ||
    message.includes("RESOURCE_EXHAUSTED")
  );
}

async function generateWithRetry(
  ai: GoogleGenAI,
  systemInstruction: string,
  userPrompt: string,
  seed?: number
): Promise<string> {
  const maxAttempts = 3;
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const response = await ai.models.generateContent({
        model: MODEL_NAME,
        contents: userPrompt,
        config: {
          systemInstruction,
          responseMimeType: "application/json",
          responseSchema: REPHRASE_SCHEMA,
          ...(typeof seed === "number" ? { seed } : {}),
        },
      });

      const text = response.text;
      if (!text) {
        throw new Error("Model returned an empty response.");
      }
      return text;
    } catch (err) {
      lastError = err;
      if (attempt < maxAttempts && isTransientError(err)) {
        const delayMs = 400 * Math.pow(2, attempt - 1);
        await new Promise((resolve) => setTimeout(resolve, delayMs));
        continue;
      }
      break;
    }
  }

  throw lastError;
}

function mapGeminiError(err: unknown): { status: number; message: string } {
  const raw = err instanceof Error ? err.message : "Unexpected server error";

  if (raw.includes("GEMINI_API_KEY is not configured")) {
    return { status: 500, message: raw };
  }
  if (raw.includes("free tier is not available") || raw.includes("set up billing")) {
    return {
      status: 500,
      message:
        "The Gemini API free tier is not available for this project. Please enable billing or select a paid API key in Settings > Secrets.",
    };
  }
  if (raw.includes("API_KEY_INVALID") || raw.includes("PERMISSION_DENIED") || raw.includes("403")) {
    return {
      status: 500,
      message:
        "Gemini API access was denied. Please verify your GEMINI_API_KEY in the Settings > Secrets panel.",
    };
  }
  if (raw.includes("RESOURCE_EXHAUSTED") || raw.includes("429")) {
    return {
      status: 429,
      message:
        "Rate limit reached on the AI service. Please wait a few seconds and try again.",
    };
  }
  if (raw.includes("NOT_FOUND") || raw.includes("404")) {
    return {
      status: 500,
      message: "Configured Gemini model (gemini-4-hydrogen) was unavailable.",
    };
  }
  return {
    status: 500,
    message: raw || "Failed to rewrite text. Please try again.",
  };
}

async function startServer(): Promise<void> {
  const app = express();
  app.use(express.json({ limit: "1mb" }));

  app.get("/api/health", (_req: Request, res: Response) => {
    res.json({ ok: true });
  });

  app.post("/api/rephrase", async (req: Request, res: Response) => {
    try {
      const body = (req.body ?? {}) as RephraseBody;
      const text = typeof body.text === "string" ? body.text.trim() : "";

      if (!text) {
        res.status(400).json({ error: "Please enter some text to rewrite." });
        return;
      }

      if (text.length > 12000) {
        res.status(400).json({
          error: "Input text exceeds the 12,000 character limit. Please shorten your draft.",
        });
        return;
      }

      const style =
        typeof body.style === "string" && VALID_STYLES.has(body.style)
          ? body.style
          : "Professional";
      const tone =
        typeof body.tone === "string" && VALID_TONES.has(body.tone)
          ? body.tone
          : "Neutral";
      const length =
        typeof body.length === "string" && VALID_LENGTHS.has(body.length)
          ? body.length
          : "Balanced";
      const clarity =
        typeof body.clarity === "string" && VALID_CLARITIES.has(body.clarity)
          ? body.clarity
          : "Clear & Direct";
      const fixGrammar = body.fixGrammar !== false;
      const preserveMeaning = body.preserveMeaning !== false;
      const variationSeed =
        typeof body.variationSeed === "number" && Number.isFinite(body.variationSeed)
          ? Math.floor(body.variationSeed)
          : undefined;

      const lengthGuidance =
        length === "Shorter"
          ? "Tighten and condense the text by roughly 20-35% while retaining every essential fact."
          : length === "Expanded"
            ? "Elaborate modestly (roughly 20-35% longer) with smoother transitions and fuller articulation without inventing unsupported facts."
            : "Maintain a similar overall length to the original draft.";

      const clarityGuidance =
        clarity === "Simplified"
          ? "Use plain, accessible vocabulary, short crisp sentences, and remove jargon."
          : clarity === "Clear & Direct"
            ? "Favor active voice, concrete verbs, and direct sentence structure for effortless readability."
            : "Refine flow naturally while respecting the complexity of the original phrasing.";

      const systemInstruction = [
        "You are RephrasePro AI, an expert editorial rewriter and prose stylist.",
        "Rewrite the user's text to match the requested Style, Tone, Length, and Clarity.",
        preserveMeaning
          ? "CRITICAL: Preserve 100% of the original meaning, facts, names, dates, numbers, commitments, and intent. Never fabricate facts or omit key details."
          : "Keep the core message intact while allowing expressive stylistic reinterpretation.",
        fixGrammar
          ? "Correct all grammar, spelling, punctuation, subject-verb agreement, and awkward syntax issues, and document the key fixes in grammarFixes."
          : "Focus primarily on style and tone transformation while keeping mechanical edits minimal.",
        "Provide 3 to 5 high-value wordingSuggestions where each targetPhrase appears verbatim inside your rewrittenText so the user can click to swap in the suggestedAlternative cleanly.",
        "Preserve paragraph breaks when the original text has multiple paragraphs.",
        "Return strictly valid JSON matching the requested schema.",
      ].join(" ");

      const userPrompt = [
        `Target Style: ${style}`,
        `Target Tone: ${tone}`,
        `Target Length: ${length} (${lengthGuidance})`,
        `Clarity Level: ${clarity} (${clarityGuidance})`,
        `Fix Grammar & Mechanics: ${fixGrammar ? "Yes" : "Minimal"}`,
        `Strict Meaning Preservation: ${preserveMeaning ? "Yes" : "Moderate"}`,
        variationSeed !== undefined
          ? `Variation Request: Produce a fresh, distinct phrasing variation (seed ${variationSeed}).`
          : "",
        "",
        "Original Text:",
        `"""`,
        text,
        `"""`,
      ]
        .filter(Boolean)
        .join("\n");

      try {
        const ai = getGeminiClient();
        const rawJson = await generateWithRetry(
          ai,
          systemInstruction,
          userPrompt,
          variationSeed
        );

        const cleanedJson = rawJson
          .trim()
          .replace(/^```(?:json)?\s*/i, "")
          .replace(/\s*```$/i, "");

        const parsed = JSON.parse(cleanedJson) as Record<string, unknown>;
        const rewrittenText =
          typeof parsed.rewrittenText === "string" ? parsed.rewrittenText.trim() : "";
        if (!rewrittenText) {
          throw new Error("The AI model returned an empty rewrite.");
        }

        res.json({
          rewrittenText,
          meaningPreservationNote:
            typeof parsed.meaningPreservationNote === "string"
              ? parsed.meaningPreservationNote
              : "",
          toneSummary: typeof parsed.toneSummary === "string" ? parsed.toneSummary : "",
          grammarFixes: Array.isArray(parsed.grammarFixes) ? parsed.grammarFixes : [],
          wordingSuggestions: Array.isArray(parsed.wordingSuggestions)
            ? parsed.wordingSuggestions
            : [],
          isDemoMode: false,
        });
      } catch (aiErr) {
        const mapped = mapGeminiError(aiErr);
        console.warn(`Falling back to Demo Mode: ${mapped.message}`);
        const demoResult = generateDemoRephrase({
          text,
          style: style as "Professional" | "Casual" | "Academic" | "Creative" | "Friendly",
          tone: tone as "Neutral" | "Confident" | "Diplomatic" | "Empathetic" | "Persuasive",
          length: length as "Shorter" | "Balanced" | "Expanded",
          clarity: clarity as "Standard" | "Clear & Direct" | "Simplified",
          fixGrammar,
          preserveMeaning,
          variationSeed,
        });
        res.json(demoResult);
      }
    } catch (err) {
      const mapped = mapGeminiError(err);
      res.status(mapped.status).json({ error: mapped.message });
    }
  });

  app.all("/api/*", (_req: Request, res: Response) => {
    res.status(400).json({ error: "Unknown API endpoint." });
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, "dist");
    if (fs.existsSync(distPath)) {
      app.use(express.static(distPath));
      app.get("*", (_req: Request, res: Response) => {
        res.sendFile(path.join(distPath, "index.html"));
      });
    }
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`RephrasePro AI server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
