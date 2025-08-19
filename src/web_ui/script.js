function runPrediction() {
    window.pywebview.api.run_model().then(result => {
        document.getElementById('result').innerText = '预测结果: ' + result;
    });
}
