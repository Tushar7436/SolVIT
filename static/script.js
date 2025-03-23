document.getElementById('analyzeButton').addEventListener('click', function() {
    const queryInput = document.getElementById('queryInput').value;

    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: queryInput }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('results').innerHTML = `
            <h2>Analysis Results:</h2>
            <p><strong>Spam Detection:</strong> ${data.spam}</p>
            <p><strong>Classification:</strong> ${data.classification}</p>
            <p><strong>Analysis:</strong> ${data.analysis}</p>
            <p><strong>Priority:</strong> ${data.priority}</p>
        `;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('results').innerHTML = `
            <p>Error analyzing query: ${error.message}</p>
        `;
    });
});