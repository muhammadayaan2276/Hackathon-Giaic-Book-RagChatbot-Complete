const testData = {
  query: "What is RAG?",
  selected_text: null
};

async function testAPI() {
  try {
    console.log("Testing the RAG Chatbot API...");
    const response = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testData)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("API Response:", data);
    console.log("Answer:", data.answer);
  } catch (error) {
    console.error("Error testing API:", error);
  }
}

testAPI();