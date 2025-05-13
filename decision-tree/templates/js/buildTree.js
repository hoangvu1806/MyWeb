document.getElementById("csvFile").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) {
        showNotification("No file selected!", "warning");
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const csvContent = e.target.result;
        const rows = csvContent.split("\n").map((row) => row.trim());
        if (rows.length === 0) {
            showNotification("The CSV file is empty!", "warning");
            return;
        }
        const headers = rows[0].split(",").map((header) => header.trim());
        const targetColumnSelect = document.getElementById("targetColumn");
        targetColumnSelect.innerHTML = "";

        headers.forEach((header) => {
            const option = document.createElement("option");
            option.value = header;
            option.textContent = header;
            targetColumnSelect.appendChild(option);
        });

        showNotification("Target column options updated!", "success");
    };

    reader.onerror = (error) => {
        console.error("Error reading file:", error);
        showNotification("Error reading the CSV file!", "error");
    };

    reader.readAsText(file);
});
document.getElementById("buildTree").addEventListener("click", async () => {
    const csvFile = document.getElementById("csvFile").files[0];
    const targetColumn = document.getElementById("targetColumn").value.trim();
    const dropColumns = document
        .getElementById("dropColumn")
        .value.trim()
        .split(",")
        .map((col) => col.trim())
        .filter((col) => col !== "");
    console.log("DropColumns: " + dropColumns);
    const maxDepth = document.getElementById("maxDepth").value.trim();
    const minSplit = document.getElementById("minSplit").value.trim();
    const criterionElement = document.getElementById("criterion");

    if (!criterionElement) {
        console.error("Criterion element not found!");
        return;
    }
    const criterion = criterionElement.value.trim();

    if (!csvFile) {
        showNotification("Please choose a CSV file", "warning");
        return;
    }
    if (targetColumn === "") {
        showNotification("Please choose a target column", "warning");
        return;
    }

    const formData = new FormData();
    formData.append("csvFile", csvFile);
    formData.append("targetColumn", targetColumn);
    formData.append("dropColumn", JSON.stringify(dropColumns));
    formData.append("maxDepth", maxDepth);
    formData.append("minSplit", minSplit);
    formData.append("criterion", criterion);

    // Debug: Print FormData entries
    for (const [key, value] of formData.entries()) {
        console.log(`${key}:`, value);
    }
    // Send to server
    try {
        const response = await fetch(
            "https://hoangvu.id.vn/decision-tree/build-tree",
            {
                method: "POST",
                body: formData,
            }
        );
        if (!response.ok) {
            showNotification(`HTTP error! status: ${response.status}`, "error");
        }
        const result = await response.json();
        console.log("Server Response:", result);
        buildTree(result.tree);
        showNotification("Built tree successfully", "success");
        showResults(result.metric);
    } catch (error) {
        console.error("Error sending data:", error);
        showNotification(error, "error");
    }
});

function showNotification(message, status) {
    const notification = document.getElementById("notification");

    notification.className = "notification";
    notification.classList.add(status, "show");

    notification.textContent = message;

    // Hide the notification after 2,5s
    setTimeout(() => {
        notification.classList.remove("show");
    }, 2500);
}

function buildTree(data) {
    const root = document.getElementById("tree");
    root.innerHTML = ""; // Clear existing tree
    const classColors = {}; // Object to store colors for each class
    const colors = [
        "#e57373",
        "#81c784",
        "#64b5f6",
        "#ffb74d",
        "#4db6ac",
        "#ba68c8",
        "#ffd54f",
        "#7986cb",
        "#a1887f",
        "#90a4ae",
    ]; // Array of colors to use

    let colorIndex = 0; // Track which color to use next

    // Recursive function to create tree nodes
    function createTreeNode(node) {
        const treeNode = document.createElement("li");

        if (node.label !== undefined && node.label !== null) {
            // leaf node
            const leaf = document.createElement("a");
            leaf.textContent = `Class: ${node.label}`;
            leaf.classList.add("leaf-node");

            if (!classColors[node.label]) {
                classColors[node.label] = colors[colorIndex];
                colorIndex = (colorIndex + 1) % colors.length;
            }
            leaf.style.backgroundColor = classColors[node.label];
            treeNode.appendChild(leaf);
        } else {
            const decision = document.createElement("a");
            const operator = typeof node.value === "number" ? "≤" : "≠";
            decision.innerHTML = `
            <span id="nodeFeature">${node.feature}</span>
            <small>${operator} ${node.value}</small>
        `;
            treeNode.appendChild(decision);

            const branches = document.createElement("ul");

            // Left branch
            if (node.left) {
                const leftNode = createTreeNode(node.left);

                // add label "True"
                const trueLabel = document.createElement("span");
                trueLabel.textContent = "true";
                trueLabel.classList.add("branch-label", "left-branch-label");
                leftNode.appendChild(trueLabel);

                branches.appendChild(leftNode);
            }

            // Right branch
            if (node.right) {
                const rightNode = createTreeNode(node.right);

                // add label "False"
                const falseLabel = document.createElement("span");
                falseLabel.textContent = "false";
                falseLabel.classList.add("branch-label", "right-branch-label");
                rightNode.appendChild(falseLabel);

                branches.appendChild(rightNode);
            }
            treeNode.appendChild(branches);
        }
        return treeNode;
    }

    const tree = document.createElement("ul");
    tree.classList.add("tree");
    tree.appendChild(createTreeNode(data));
    root.appendChild(tree);
}
function showResults(metric) {
    const resultsDiv = document.getElementById("results");
    resultsDiv.classList.remove("hidden"); // Hiển thị khung kết quả
    const metrics = `
        <div class="grid-container">
            <div>Accuracy:</div>
            <div>${(metric.accuracy * 100).toFixed(2)}%</div>
            <div>Precision:</div>
            <div>${(metric.precision * 100).toFixed(2)}%</div>
            <div>Recall:</div>
            <div>${(metric.recall * 100).toFixed(2)}%</div>
            <div>F1 Score:</div>
            <div>${(metric.f1_score * 100).toFixed(2)}%</div>
        </div>`;
    resultsDiv.innerHTML = metrics;
}
