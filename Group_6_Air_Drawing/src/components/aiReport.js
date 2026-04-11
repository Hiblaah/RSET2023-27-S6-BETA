export async function generateAIReport(sessionData) {
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${import.meta.env.VITE_OPENAI_KEY}`,
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            "You are a supportive AI helping children with dysgraphia practice writing. Always be encouraging."
        },
        {
          role: "user",
          content: `
Here are the results of a writing practice session.

DATA:
${JSON.stringify(sessionData, null, 2)}

Analyze:
- Which letters were hardest
- Which words had issues
- Which letters inside words caused difficulty
- Which items were skipped

Give a short encouraging report telling them they are improving and will succeed with practice.
`
        }
      ],
      temperature: 0.7
    })
  });

  const data = await response.json();
  return data.choices[0].message.content;
}