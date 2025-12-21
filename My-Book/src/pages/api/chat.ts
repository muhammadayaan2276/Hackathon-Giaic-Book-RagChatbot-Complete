import { bookData } from '../../data/bookData';
import type { NextApiRequest, NextApiResponse } from 'next';

// Define types for our API
type Data = {
  reply: string;
};

// Calculate similarity between two strings using a simple token-based approach
function calculateSimilarity(userInput: string, question: string): number {
  // Convert both strings to lowercase and split into tokens
  const userTokens = userInput.toLowerCase().split(/\W+/).filter(token => token.length > 0);
  const questionTokens = question.toLowerCase().split(/\W+/).filter(token => token.length > 0);
  
  // Count matching tokens
  let matches = 0;
  for (const token of userTokens) {
    if (questionTokens.includes(token)) {
      matches++;
    }
  }
  
  // Return similarity ratio (higher of the two possible ratios to account for different lengths)
  const ratio1 = matches / userTokens.length;
  const ratio2 = matches / questionTokens.length;
  return Math.max(ratio1, ratio2);
}

export default function handler(req: NextApiRequest, res: NextApiResponse<Data>) {
  if (req.method !== 'POST') {
    res.status(405).json({ reply: 'Method not allowed' });
    return;
  }

  const { message } = req.body;

  if (!message) {
    res.status(400).json({ reply: 'Message is required' });
    return;
  }

  // Find the best matching question in our dataset
  let bestMatch = null;
  let bestScore = 0;

  for (const item of bookData) {
    const score = calculateSimilarity(message, item.question);
    
    // If this score is better than our current best, update the best match
    if (score > bestScore) {
      bestScore = score;
      bestMatch = item;
    }
  }

  // Check if we found a match with sufficient similarity (at least 50%)
  if (bestMatch && bestScore >= 0.5) {
    res.status(200).json({ reply: bestMatch.answer });
  } else {
    res.status(200).json({ reply: 'Answer not found in my book.' });
  }
}