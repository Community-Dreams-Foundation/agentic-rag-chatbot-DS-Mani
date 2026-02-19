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

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendBtn.click();
  }
});
