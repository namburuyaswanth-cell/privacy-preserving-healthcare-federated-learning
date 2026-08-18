const API = "/api";

async function getJSON(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  return { response, data };
}

async function loadHealth() {
  const pill = document.getElementById("serverStatus");
  try {
    const { data } = await getJSON(`${API}/health`);
    pill.textContent = data.status === "online" ? "● Server online" : "Server offline";
  } catch (error) {
    pill.textContent = "● Server unavailable";
  }
}

async function loadProject() {
  try {
    const { data } = await getJSON(`${API}/project`);
    const metrics = data.reported_metrics;
    document.getElementById("accuracy").textContent = metrics.accuracy;
    document.getElementById("precision").textContent = metrics.precision;
    document.getElementById("recall").textContent = metrics.recall;
    document.getElementById("f1").textContent = metrics.f1_score;
  } catch (error) {
    console.error(error);
  }
}

async function loadClients() {
  const container = document.getElementById("clients");

  try {
    const { data } = await getJSON(`${API}/federated/status`);

    container.innerHTML = data.clients.map(client => `
      <div class="client">
        <strong>${client.id}</strong>
        <span class="client-status">● ${client.status}</span>
      </div>
    `).join("");
  } catch (error) {
    container.innerHTML = "<p>Unable to load client status.</p>";
  }
}

document.getElementById("authForm").addEventListener("submit", async (event) => {
  event.preventDefault();

  const deviceId = document.getElementById("deviceId").value.trim();
  const result = document.getElementById("authResult");
  result.textContent = "Authenticating...";

  try {
    const { data } = await getJSON(`${API}/authenticate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_id: deviceId })
    });

    result.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    result.textContent = `Authentication request failed: ${error}`;
  }
});

document.getElementById("predictionForm").addEventListener("submit", async (event) => {
  event.preventDefault();

  const input = document.getElementById("imageInput");
  const result = document.getElementById("predictionResult");

  if (!input.files.length) {
    result.textContent = "Please select an image.";
    return;
  }

  const formData = new FormData();
  formData.append("image", input.files[0]);

  result.textContent = "Uploading image and checking model...";

  try {
    const { data } = await getJSON(`${API}/predict`, {
      method: "POST",
      body: formData
    });

    if (data.prediction) {
      result.innerHTML = `
        <strong>Prediction: ${data.prediction}</strong><br>
        Probability: ${data.probability}
      `;
    } else {
      result.textContent = data.message || "No prediction returned.";
    }
  } catch (error) {
    result.textContent = `Prediction request failed: ${error}`;
  }
});

loadHealth();
loadProject();
loadClients();
