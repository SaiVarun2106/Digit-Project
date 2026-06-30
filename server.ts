import express from "express";
import path from "path";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

// Initialize Express
const app = express();
app.use(express.json());

const PORT = 3000;

// Initialize Google Gemini SDK
// Lazy initialization or fallback in case GEMINI_API_KEY is not defined yet
const getGeminiClient = () => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.warn("WARNING: GEMINI_API_KEY environment variable is not set. Using mock fallbacks.");
    return null;
  }
  return new GoogleGenAI({
    apiKey: apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      }
    }
  });
};

// API Endpoint to Generate 10 Questions based on Board, Class, Subject, Chapter, Topics, and Difficulty
app.post("/api/assessment/generate", async (req, res) => {
  const { board, grade, subject, chapter, topics, difficulty } = req.body;

  if (!board || !grade || !subject || !chapter) {
    return res.status(400).json({ error: "Missing required fields (board, grade, subject, chapter)" });
  }

  const ai = getGeminiClient();
  if (!ai) {
    // Return high-fidelity fallback Mock Questions if API Key is not set yet
    return res.json({
      questions: getFallbackQuestions(subject, chapter, difficulty)
    });
  }

  try {
    const prompt = `You are an expert curriculum developer and academic examiner.
Generate exactly 10 academic questions based on the following school syllabus context:
- Education Board: ${board}
- Grade/Class: ${grade}
- Subject/Module: ${subject}
- Chapter: ${chapter}
- Specific Topics of focus: ${topics || "general topics under chapter"}
- Difficulty Level: ${difficulty || "Medium"}

You MUST include a variety of question types such as:
1. MCQ (Multiple Choice Questions) - must have options and correct answer
2. Short Answer - open ended question with suggested answer
3. Conceptual - questions checking deep understanding of concepts with suggested explanation
4. True or False - must have True/False options and correct answer

Format your output STRICTLY as a JSON array of objects. Each object should have the following structure:
{
  "id": number (1 to 10),
  "question": "The question text",
  "type": "mcq" | "short_answer" | "conceptual" | "true_false",
  "options": ["Option A", "Option B", "Option C", "Option D"] (only for mcq and true_false types, otherwise empty or null. For true_false, use ["True", "False"]),
  "correctAnswer": "The correct answer (e.g. the specific option for mcq/true_false, or an ideal brief answer explanation for other types)"
}

Do not include any markdown backticks, HTML tags, explanations, or text outside the JSON array. Start directly with [ and end with ].`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              id: { type: Type.INTEGER },
              question: { type: Type.STRING },
              type: { type: Type.STRING },
              options: {
                type: Type.ARRAY,
                items: { type: Type.STRING }
              },
              correctAnswer: { type: Type.STRING }
            },
            required: ["id", "question", "type", "correctAnswer"]
          }
        }
      }
    });

    const text = response.text;
    if (!text) {
      throw new Error("No response text received from Gemini API");
    }

    const parsedQuestions = JSON.parse(text.trim());
    return res.json({ questions: parsedQuestions });

  } catch (error: any) {
    console.error("Gemini Generation Error:", error);
    // Return high-quality fallbacks on failure
    return res.json({
      error: error.message,
      questions: getFallbackQuestions(subject, chapter, difficulty)
    });
  }
});

// API Endpoint to Evaluate Assessment Answers using Gemini AI
app.post("/api/assessment/evaluate", async (req, res) => {
  const { grade, subject, chapter, questions, answers } = req.body;

  if (!questions || !answers) {
    return res.status(400).json({ error: "Missing questions or answers for evaluation" });
  }

  const ai = getGeminiClient();
  if (!ai) {
    return res.json({
      evaluation: getFallbackEvaluation(questions, answers, subject)
    });
  }

  try {
    // Map questions with answers
    const questionsAndAnswers = questions.map((q: any, idx: number) => ({
      id: q.id,
      question: q.question,
      type: q.type,
      correctAnswer: q.correctAnswer,
      studentAnswer: answers[idx] || "(No Answer Provided)"
    }));

    const prompt = `You are an expert AI tutor evaluating a student's completed academic assessment.
Below are the details of the assessment:
- Grade/Class Level: ${grade || "Class 10"}
- Subject: ${subject || "General"}
- Chapter/Topic: ${chapter || "General Concepts"}

Questions and Answers:
${JSON.stringify(questionsAndAnswers, null, 2)}

Please evaluate the student's responses carefully.
Consider correctness, conceptual understanding, explanation depth, and grade level.
Assess how well they understand the core concept behind each question.
Provide encouragement and clear suggestions.

Format your output STRICTLY as a JSON object with the following structure:
{
  "score": "X/10" (numerical score out of total questions, e.g. "8/10"),
  "percentage": number (score converted to 0-100 percentage, e.g. 80),
  "feedback": "Overall comprehensive, encouraging feedback highlighting strengths and general advice for parents to support.",
  "weakConcepts": ["Concept A", "Concept B"] (specific subtopics or concepts where the student showed misunderstanding or gaps),
  "recommendations": {
    "topicsToRevise": ["Topic 1", "Topic 2"],
    "practiceQuestions": ["Practice Question 1", "Practice Question 2"],
    "revisionPlan": "A brief step-by-step revision strategy.",
    "dailyGoals": "A few short-term actionable goals (e.g., spend 15 minutes reviewing active formulas)."
  }
}

Do not include any markdown backticks, HTML tags, explanations, or text outside the JSON object. Start directly with { and end with }.`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            score: { type: Type.STRING },
            percentage: { type: Type.INTEGER },
            feedback: { type: Type.STRING },
            weakConcepts: {
              type: Type.ARRAY,
              items: { type: Type.STRING }
            },
            recommendations: {
              type: Type.OBJECT,
              properties: {
                topicsToRevise: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING }
                },
                practiceQuestions: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING }
                },
                revisionPlan: { type: Type.STRING },
                dailyGoals: { type: Type.STRING }
              },
              required: ["topicsToRevise", "practiceQuestions", "revisionPlan", "dailyGoals"]
            }
          },
          required: ["score", "percentage", "feedback", "weakConcepts", "recommendations"]
        }
      }
    });

    const text = response.text;
    if (!text) {
      throw new Error("No evaluation response text received from Gemini API");
    }

    const evaluation = JSON.parse(text.trim());
    return res.json({ evaluation });

  } catch (error: any) {
    console.error("Gemini Evaluation Error:", error);
    return res.json({
      error: error.message,
      evaluation: getFallbackEvaluation(questions, answers, subject)
    });
  }
});


// High-Fidelity Fallback generators when API keys are not ready or if rate-limited
function getFallbackQuestions(subject: string, chapter: string, difficulty: string) {
  const normSubject = (subject || "").toLowerCase();
  
  if (normSubject.includes("math") || normSubject.includes("algebra")) {
    return [
      {
        id: 1,
        question: `Find the value of x that satisfies x² - 5x + 6 = 0 for the chapter: ${chapter}.`,
        type: "mcq",
        options: ["x = 2, 3", "x = -2, -3", "x = 1, 5", "x = 0, 6"],
        correctAnswer: "x = 2, 3"
      },
      {
        id: 2,
        question: "An Arithmetic Sequence has first term a = 3 and common difference d = 2. What is the 5th term?",
        type: "mcq",
        options: ["9", "11", "13", "15"],
        correctAnswer: "11"
      },
      {
        id: 3,
        question: "Find the slope of a line perpendicular to y = 2x + 7.",
        type: "mcq",
        options: ["-1/2", "2", "-2", "1/2"],
        correctAnswer: "-1/2"
      },
      {
        id: 4,
        question: "A quadratic equation always has exactly two real distinct roots.",
        type: "true_false",
        options: ["True", "False"],
        correctAnswer: "False"
      },
      {
        id: 5,
        question: "State the quadratic formula used to solve any equation ax² + bx + c = 0.",
        type: "short_answer",
        correctAnswer: "x = (-b ± √(b² - 4ac)) / 2a"
      },
      {
        id: 6,
        question: "Explain conceptually why a vertical line has an undefined slope.",
        type: "conceptual",
        correctAnswer: "Slope is rise over run. A vertical line has zero run (no horizontal change), meaning you divide by zero, which is mathematically undefined."
      },
      {
        id: 7,
        question: "Solve the linear system of equations: x + y = 5, x - y = 1. What is x?",
        type: "mcq",
        options: ["x = 3", "x = 2", "x = 4", "x = 1"],
        correctAnswer: "x = 3"
      },
      {
        id: 8,
        question: "The sum of the angles in any planar triangle is always 180 degrees.",
        type: "true_false",
        options: ["True", "False"],
        correctAnswer: "True"
      },
      {
        id: 9,
        question: "What is the common ratio in the geometric sequence: 2, 6, 18, 54...?",
        type: "mcq",
        options: ["2", "3", "4", "6"],
        correctAnswer: "3"
      },
      {
        id: 10,
        question: "Describe what a function's domain represents in a coordinate plane.",
        type: "conceptual",
        correctAnswer: "The domain represents the complete set of all possible input values (usually x-values) for which the function is defined and produces real numbers."
      }
    ];
  } else if (normSubject.includes("science") || normSubject.includes("biological")) {
    return [
      {
        id: 1,
        question: `Which of the following cellular organelle is known as the powerhouse of the cell for ${chapter}?`,
        type: "mcq",
        options: ["Nucleus", "Ribosome", "Mitochondria", "Lysosome"],
        correctAnswer: "Mitochondria"
      },
      {
        id: 2,
        question: "What is the main chemical product of photosynthesis that plants use for food?",
        type: "mcq",
        options: ["Oxygen", "Glucose", "Carbon Dioxide", "Water"],
        correctAnswer: "Glucose"
      },
      {
        id: 3,
        question: "Plant cells contain a rigid cell wall, whereas animal cells do not.",
        type: "true_false",
        options: ["True", "False"],
        correctAnswer: "True"
      },
      {
        id: 4,
        question: "State Newton's Second Law of Motion in terms of Force, Mass, and Acceleration.",
        type: "short_answer",
        correctAnswer: "Force equals mass times acceleration (F = ma)."
      },
      {
        id: 5,
        question: "Explain the difference between a physical change and a chemical change.",
        type: "conceptual",
        correctAnswer: "A physical change alters the state or appearance without changing chemical identity (like ice melting). A chemical change forms entirely new chemical substances with new bonds (like wood burning)."
      },
      {
        id: 6,
        question: "What is the pH level of pure distilled water at room temperature?",
        type: "mcq",
        options: ["5.0", "7.0", "9.0", "14.0"],
        correctAnswer: "7.0"
      },
      {
        id: 7,
        question: "Sound waves travel faster in a vacuum than they do in solid iron bars.",
        type: "true_false",
        options: ["True", "False"],
        correctAnswer: "False"
      },
      {
        id: 8,
        question: "Which blood cells are primarily responsible for fighting infections and pathogens?",
        type: "mcq",
        options: ["Red Blood Cells", "White Blood Cells", "Platelets", "Plasma Cells"],
        correctAnswer: "White Blood Cells"
      },
      {
        id: 9,
        question: "Define the term 'Genotype' in modern genetic biology.",
        type: "short_answer",
        correctAnswer: "The genotype is the unique genetic constitution or allele makeup of an individual organism."
      },
      {
        id: 10,
        question: "Describe conceptually why oil floats on water instead of mixing with it.",
        type: "conceptual",
        correctAnswer: "Oil floats because it is less dense than water. It does not mix because oil is nonpolar (hydrophobic), while water is a highly polar solvent, meaning they cannot form stable intermolecular bonds."
      }
    ];
  } else {
    // General high-quality humanities/default fallback
    return [
      {
        id: 1,
        question: `Who is the protagonist in the classic novel or literature unit: ${chapter}?`,
        type: "mcq",
        options: ["The Antagonist", "The Narrator", "The Lead Hero/Protagonist", "The Focal Character"],
        correctAnswer: "The Lead Hero/Protagonist"
      },
      {
        id: 2,
        question: "A metaphor is a direct comparison using the terms 'like' or 'as'.",
        type: "true_false",
        options: ["True", "False"],
        correctAnswer: "False"
      },
      {
        id: 3,
        question: "What is the primary theme or moral conflict highlighted in this literary chapter?",
        type: "short_answer",
        correctAnswer: "The central conflict pits individual morality against societal pressure."
      },
      {
        id: 4,
        question: "Describe what 'alliteration' is and provide a simple example.",
        type: "conceptual",
        correctAnswer: "Alliteration is the repetition of the same consonant sounds at the beginning of adjacent or closely connected words. Example: 'Peter Piper picked a peck...'"
      },
      {
        id: 5,
        question: "Who wrote the historical tragedy 'Romeo and Juliet'?",
        type: "mcq",
        options: ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
        correctAnswer: "William Shakespeare"
      },
      {
        id: 6,
        question: "The Industrial Revolution first began in the United States in the late 17th century.",
        type: "true_false",
        options: ["True", "False"],
        correctAnswer: "False"
      },
      {
        id: 7,
        question: "Which rhetorical appeal relies on establishing the character and credibility of the speaker?",
        type: "mcq",
        options: ["Pathos", "Logos", "Ethos", "Kairos"],
        correctAnswer: "Ethos"
      },
      {
        id: 8,
        question: "Define 'onomatopoeia' in poetic literature analysis.",
        type: "short_answer",
        correctAnswer: "It is a word that phonetically mimics or resembles the sound it describes, like 'buzz', 'sizzle', or 'bang'."
      },
      {
        id: 9,
        question: "Describe conceptually how setting can influence a story's character development.",
        type: "conceptual",
        correctAnswer: "Setting shapes a character's values, obstacles, and opportunities. A harsh winter landscape can force a character to discover inner resilience or make difficult sacrifices."
      },
      {
        id: 10,
        question: "A sonnet is a traditional lyric poem composed of exactly 14 rhyming lines.",
        type: "true_false",
        options: ["True", "False"],
        correctAnswer: "True"
      }
    ];
  }
}

function getFallbackEvaluation(questions: any[], answers: any, subject: string) {
  let correctCount = 0;
  questions.forEach((q, idx) => {
    const studentAns = (answers[idx] || "").trim().toLowerCase();
    const correctAns = q.correctAnswer.trim().toLowerCase();
    
    if (q.type === 'mcq' || q.type === 'true_false') {
      if (studentAns === correctAns) {
        correctCount++;
      }
    } else {
      // For open-ended/conceptual, if they wrote something reasonable
      if (studentAns.length > 5) {
        correctCount++;
      }
    }
  });

  const percentage = Math.round((correctCount / questions.length) * 100);

  let weakConcepts = [];
  if (percentage < 70) {
    weakConcepts = [
      "Conceptual explanation clarity",
      "Advanced applications of rules",
      "Detail tracking in complex terms"
    ];
  } else if (percentage < 90) {
    weakConcepts = ["Boundary conditions", "Finer technical terminology"];
  } else {
    weakConcepts = ["None - Excellent understanding! Just keep practicing for speed."];
  }

  return {
    score: `${correctCount}/${questions.length}`,
    percentage,
    feedback: `Great attempt! The student answered ${correctCount} questions correctly. They displayed strong engagement. Areas for minor adjustment include conceptual reasoning on complex parts. Parents should guide them with regular 10-minute active review sessions.`,
    weakConcepts,
    recommendations: {
      topicsToRevise: [
        "Core formulas and derivations",
        "Contextual applications"
      ],
      practiceQuestions: [
        "Solve 3 practice questions on this topic every afternoon.",
        "Review key definitions on index cards."
      ],
      revisionPlan: "Spend 20 minutes reviewing incorrect answers, write down the correct formulas, and attempt similar problems from the syllabus workbook.",
      dailyGoals: "Devote 15 minutes each morning to active recall of yesterday's weak topics."
    }
  };
}


// Start Vite development server or static serving
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    // Serve production static build
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
