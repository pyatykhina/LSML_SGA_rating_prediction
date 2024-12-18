document.getElementById('form').addEventListener('submit', function (e) {
    e.preventDefault();
    const inputData = document.getElementById('inputData').value;

    fetch('http://api:8888/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputData),
    })
    .then(response => {
        response.json()
    })
    .then(data => {
        document.getElementById('prediction').innerText = `Prediction: ${data.prediction}`;
    });
});