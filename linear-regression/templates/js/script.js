// JavaScript functions for handling UI interactions
function chooseCSV() {
    alert("Chọn file CSV để tải lên.");
}

function chooseTargetColumn() {
    alert("Chọn cột mục tiêu để huấn luyện.");
}

function startTraining() {
    const learningRate = document.getElementById("learningRate").value;
    const epochs = document.getElementById("epochs").value;
    alert(
        `Bắt đầu training với Learning Rate: ${learningRate}, Epochs: ${epochs}`
    );
}

// WebSocket hoặc AJAX để nhận dữ liệu huấn luyện từ server
// Ví dụ WebSocket: kết nối tới server và hiển thị dữ liệu vào Console Area
let websocket = new WebSocket("ws://localhost:8000/ws/progress");

websocket.onopen = function () {
    console.log("Connected to WebSocket");
};

websocket.onmessage = function (event) {
    const data = JSON.parse(event.data);
    const consoleArea = document.querySelector(".console-area");
    const message = `Epoch: ${data.epoch}, Loss: ${data.loss}`;
    consoleArea.innerHTML += `<div>${message}</div>`;
    consoleArea.scrollTop = consoleArea.scrollHeight;
};

websocket.onclose = function () {
    console.log("Disconnected from WebSocket");
};
