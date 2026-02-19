const themeToggle = document.getElementById("themeToggle");
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const fileList = document.getElementById("fileList");
const useEmbeddings = document.getElementById("useEmbeddings");
const topK = document.getElementById("topK");
const citeK = document.getElementById("citeK");
const embedModel = document.getElementById("embedModel");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const messages = document.getElementById("messages");
const weatherState = document.getElementById("weatherState");
const weatherCity = document.getElementById("weatherCity");
const weatherCountry = document.getElementById("weatherCountry");
const weatherStart = document.getElementById("weatherStart");
const weatherEnd = document.getElementById("weatherEnd");
const weatherSandbox = document.getElementById("weatherSandbox");
const weatherBtn = document.getElementById("weatherBtn");
const weatherStatus = document.getElementById("weatherStatus");
const weatherSearchBtn = document.getElementById("weatherSearchBtn");
const weatherResults = document.getElementById("weatherResults");

function setTheme(theme) {
  document.body.setAttribute("data-theme", theme);
  themeToggle.textContent = theme === "dark" ? "Dark" : "Light";
  localStorage.setItem("theme", theme);
}

const savedTheme = localStorage.getItem("theme") || "light";
setTheme(savedTheme);

themeToggle.addEventListener("click", () => {
  const current = document.body.getAttribute("data-theme");
  setTheme(current === "dark" ? "light" : "dark");
});

function setDefaultWeatherDates() {
  if (!weatherStart || !weatherEnd) return;
  const today = new Date();
  const end = today.toISOString().slice(0, 10);
  const startDate = new Date(today);
  startDate.setDate(today.getDate() - 7);
  const start = startDate.toISOString().slice(0, 10);
  if (!weatherStart.value) weatherStart.value = start;
  if (!weatherEnd.value) weatherEnd.value = end;
}

setDefaultWeatherDates();

if (weatherState && !weatherState.value) {
  weatherState.value = "California";
}
if (weatherCity && !weatherCity.value) {
  weatherCity.value = "";
}

let selectedLocation = null;

function buildLocationQuery() {
  const state = weatherState ? weatherState.value.trim() : "";
  const city = weatherCity ? weatherCity.value.trim() : "";
  if (city && state) {
    return `${city}, ${state}`;
  }
  return city || state;
}

function appendMessage(role, text, citations = []) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const p = document.createElement("p");
  p.textContent = text;
  bubble.appendChild(p);

  if (citations.length) {
    const citeBox = document.createElement("div");
    citeBox.className = "citations";
    const title = document.createElement("h4");
    title.textContent = `Citations (${citations.length})`;
    citeBox.appendChild(title);

    citations.forEach((c) => {
      const item = document.createElement("div");
      item.className = "cite-item";
      item.textContent = `${c.source} — ${c.locator}`;
      const snippet = document.createElement("span");
      snippet.textContent = c.snippet;
      item.appendChild(snippet);
      citeBox.appendChild(item);
    });
    bubble.appendChild(citeBox);
  }

  wrapper.appendChild(bubble);
  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
}

uploadBtn.addEventListener("click", async () => {
  const files = fileInput.files;
  if (!files || files.length === 0) {
    uploadStatus.textContent = "Please select files to upload.";
    return;
  }

  uploadStatus.textContent = "Uploading and building index...";
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));
  formData.append("use_embeddings", useEmbeddings.checked);
  formData.append("embed_model", embedModel.value || "all-MiniLM-L6-v2");

  try {
    const res = await fetch("/api/ingest", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Upload failed");
    }
    uploadStatus.textContent = `Indexed ${data.ingested} file(s). Embeddings: ${data.has_embeddings}`;
    fileList.innerHTML = "";
    data.files.forEach((name) => {
      const li = document.createElement("li");
      li.textContent = name;
      fileList.appendChild(li);
    });
  } catch (err) {
    uploadStatus.textContent = err.message;
  }
});

sendBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  if (!question) return;

  appendMessage("user", question);
  questionInput.value = "";

  const formData = new FormData();
  formData.append("question", question);
  const topKVal = Math.max(parseInt(topK.value || "4", 10), 1);
  const citeKVal = Math.max(parseInt(citeK.value || "3", 10), 1);
  formData.append("top_k", Math.max(topKVal, citeKVal));
  formData.append("citations_k", citeKVal);
  formData.append("use_embeddings", useEmbeddings.checked);
  formData.append("embed_model", embedModel.value || "all-MiniLM-L6-v2");

  try {
    const res = await fetch("/api/ask", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Request failed");
    }
    appendMessage("assistant", data.answer, data.citations || []);
  } catch (err) {
    appendMessage("assistant", `Error: ${err.message}`);
  }
});

weatherBtn.addEventListener("click", async () => {
  if (!weatherStart || !weatherEnd) return;
  const query = buildLocationQuery();
  const country = weatherCountry ? weatherCountry.value.trim() : "";
  const start = weatherStart.value.trim();
  const end = weatherEnd.value.trim();
  if (!query || !start || !end) {
    weatherStatus.textContent = "Please fill state/city, start, and end.";
    return;
  }

  weatherStatus.textContent = "Fetching weather...";
  const formData = new FormData();
  if (selectedLocation) {
    formData.append("lat", selectedLocation.latitude);
    formData.append("lon", selectedLocation.longitude);
    formData.append("location_name", selectedLocation.label);
  } else {
    formData.append("city", query);
    if (country) {
      formData.append("country", country);
    }
  }
  formData.append("start", start);
  formData.append("end", end);
  formData.append("sandbox", weatherSandbox && weatherSandbox.checked ? "docker" : "none");

  try {
    const res = await fetch("/api/weather", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Weather request failed");
    }
    const stats = data.stats || {};
    const locationName = data.location_name || `${data.location?.lat}, ${data.location?.lon}`;
    const summary =
      "Weather summary (Open-Meteo)\n" +
      `Location: ${locationName}\n` +
      `Range: ${data.start_date} to ${data.end_date}\n` +
      `Mean temp: ${stats.mean}\n` +
      `Std dev: ${stats.std}\n` +
      `Min: ${stats.min}  Max: ${stats.max}\n` +
      `Missing: ${stats.missing}  Anomalies: ${stats.anomalies}`;

    weatherStatus.textContent = "Weather updated.";
    appendMessage("assistant", summary);
  } catch (err) {
    const msg = err.message || "Weather request failed";
    weatherStatus.textContent = msg;
    appendMessage("assistant", `Weather error: ${msg}`);
  }
});

weatherSearchBtn.addEventListener("click", async () => {
  if (!weatherResults) return;
  const query = buildLocationQuery();
  const country = weatherCountry ? weatherCountry.value.trim() : "";
  if (!query) {
    weatherStatus.textContent = "Enter a state or city first.";
    return;
  }

  weatherStatus.textContent = "Searching locations...";
  selectedLocation = null;
  weatherResults.innerHTML = "";

  const formData = new FormData();
  formData.append("query", query);
  if (country) {
    formData.append("country", country);
  }
  formData.append("limit", "8");

  try {
    const res = await fetch("/api/geocode", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Geocoding failed");
    }

    const results = data.results || [];
    if (!results.length) {
      weatherResults.textContent = "No matches found.";
      weatherStatus.textContent = "No matches found.";
      return;
    }

    results.forEach((item) => {
      const labelParts = [item.name];
      if (item.admin1) labelParts.push(item.admin1);
      if (item.country_code) labelParts.push(item.country_code);
      const label = labelParts.join(", ");

      const row = document.createElement("div");
      row.className = "result-item";
      row.innerHTML = `<strong>${label}</strong><span>${item.latitude}, ${item.longitude}</span>`;
      row.addEventListener("click", () => {
        selectedLocation = { ...item, label };
        Array.from(weatherResults.children).forEach((child) => child.classList.remove("selected"));
        row.classList.add("selected");
        weatherStatus.textContent = `Selected: ${label}`;
      });
      weatherResults.appendChild(row);
    });

    weatherStatus.textContent = "Select a location from the list.";
  } catch (err) {
    const msg = err.message || "Geocoding failed";
    weatherStatus.textContent = msg;
    weatherResults.textContent = msg;
  }
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendBtn.click();
  }
});
