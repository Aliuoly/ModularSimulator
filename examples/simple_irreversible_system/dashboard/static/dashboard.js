const numberFormatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });
const timeFormatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });

const charts = {
  b: null,
  flow: null,
  volume: null,
};

function formatValue(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return numberFormatter.format(value);
}

function extractNumericValue(rawValue) {
  if (rawValue === null || rawValue === undefined) {
    return NaN;
  }

  if (Array.isArray(rawValue)) {
    // Prefer the most recent entry when a controller exposes a history vector.
    return extractNumericValue(rawValue[rawValue.length - 1]);
  }

  if (typeof rawValue === "object") {
    if ("value" in rawValue) {
      return extractNumericValue(rawValue.value);
    }
    return NaN;
  }

  const numeric = Number(rawValue);
  return Number.isFinite(numeric) ? numeric : NaN;
}

function buildPoints(history) {
  return history
    .map((point) => ({
      x: point.time / 60.0,
      y: extractNumericValue(point.value),
    }))
    .filter((point) => Number.isFinite(point.y));
}

function updateAxisBounds(chart) {
  const yScale = chart.options?.scales?.y;
  if (!yScale) {
    return;
  }

  const yValues = chart.data.datasets.flatMap((dataset) =>
    (dataset.data || [])
      .map((point) => point.y)
      .filter((value) => Number.isFinite(value))
  );

  if (!yValues.length) {
    delete yScale.min;
    delete yScale.max;
    return;
  }

  const min = Math.min(...yValues);
  const max = Math.max(...yValues);
  const padding = (max - min) * 0.1 || Math.abs(max || min) * 0.1 || 1;

  yScale.min = min - padding;
  yScale.max = max + padding;
}

function initCharts() {
  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "nearest", intersect: false },
    parsing: false,
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "Time (minutes)" },
      },
    },
    plugins: {
      legend: { position: "top" },
    },
  };

  charts.b = new Chart(document.getElementById("chart-b"), {
    type: "line",
    data: { datasets: [] },
    options: {
      ...baseOptions,
      scales: {
        ...baseOptions.scales,
        y: {
          title: { display: true, text: "Concentration" },
        },
      },
    },
  });

  charts.flow = new Chart(document.getElementById("chart-flow"), {
    type: "line",
    data: { datasets: [] },
    options: {
      ...baseOptions,
      scales: {
        ...baseOptions.scales,
        y: {
          title: { display: true, text: "Flow Rate" },
        },
      },
    },
  });

  charts.volume = new Chart(document.getElementById("chart-volume"), {
    type: "line",
    data: { datasets: [] },
    options: {
      ...baseOptions,
      scales: {
        ...baseOptions.scales,
        y: {
          title: { display: true, text: "Volume" },
        },
      },
    },
  });
}

function updateCharts(data) {
  if (!charts.b || !charts.flow || !charts.volume) {
    return;
  }

  const sensorData = data.sensors || {};
  const setpoints = data.setpoints || {};
  const manipulated = data.manipulated || {};

  const bSensor = sensorData.B ? buildPoints(sensorData.B.data) : [];
  const bSetpoint = setpoints.B ? buildPoints(setpoints.B.data) : [];

  charts.b.data.datasets = [
    {
      label: "B (sensor)",
      data: bSensor,
      borderColor: "#2563eb",
      backgroundColor: "rgba(37, 99, 235, 0.2)",
      tension: 0.2,
    },
    {
      label: "B Setpoint",
      data: bSetpoint,
      borderColor: "#f97316",
      borderDash: [6, 4],
      backgroundColor: "rgba(249, 115, 22, 0.2)",
      tension: 0.2,
    },
  ];
  updateAxisBounds(charts.b);
  charts.b.update("none");

  const fInSensor = sensorData.F_in ? buildPoints(sensorData.F_in.data) : [];
  const fOutSensor = sensorData.F_out ? buildPoints(sensorData.F_out.data) : [];
  const fInCommand = manipulated.F_in ? buildPoints(manipulated.F_in.data) : [];

  charts.flow.data.datasets = [
    {
      label: "F_in (sensor)",
      data: fInSensor,
      borderColor: "#16a34a",
      backgroundColor: "rgba(22, 163, 74, 0.2)",
      tension: 0.2,
    },
    {
      label: "F_out",
      data: fOutSensor,
      borderColor: "#dc2626",
      backgroundColor: "rgba(220, 38, 38, 0.2)",
      tension: 0.2,
    },
    {
      label: "F_in (command)",
      data: fInCommand,
      borderColor: "#0ea5e9",
      borderDash: [4, 4],
      backgroundColor: "rgba(14, 165, 233, 0.2)",
      tension: 0.2,
    },
  ];
  updateAxisBounds(charts.flow);
  charts.flow.update("none");

  const volumeSensor = sensorData.V ? buildPoints(sensorData.V.data) : [];
  charts.volume.data.datasets = [
    {
      label: "V (sensor)",
      data: volumeSensor,
      borderColor: "#9333ea",
      backgroundColor: "rgba(147, 51, 234, 0.2)",
      tension: 0.2,
    },
  ];
  updateAxisBounds(charts.volume);
  charts.volume.update("none");
}

function updateTimeAndSpeed(timeSeconds, speed) {
  const timeElement = document.getElementById("sim-time");
  if (timeElement && typeof timeSeconds === "number") {
    timeElement.textContent = timeFormatter.format(timeSeconds / 60.0);
  }
  const speedElement = document.getElementById("speed-factor");
  if (speedElement && typeof speed === "number") {
    speedElement.textContent = numberFormatter.format(speed);
  }
}

function showMessage(text, type = "info") {
  const container = document.getElementById("message");
  if (!container) return;
  container.textContent = text;
  container.className = "message " + type;
  container.classList.add("visible");
  setTimeout(() => {
    container.classList.remove("visible");
  }, 4000);
}

function buildControllerCard(controller) {
  const card = document.createElement("div");
  card.className = "controller-card";
  card.dataset.cvTag = controller.cv_tag;

  const header = document.createElement("div");
  header.className = "controller-card__header";
  header.innerHTML = `<h3>${controller.cv_tag}</h3>`;
  card.appendChild(header);

  const modeWrapper = document.createElement("div");
  modeWrapper.className = "controller-card__row";
  const modeLabel = document.createElement("label");
  modeLabel.textContent = "Mode";
  const modeSelect = document.createElement("select");
  controller.available_modes.forEach((mode) => {
    const option = document.createElement("option");
    option.value = mode;
    option.textContent = mode;
    if (mode === controller.mode) {
      option.selected = true;
    }
    modeSelect.appendChild(option);
  });
  modeSelect.addEventListener("change", (event) => {
    const newMode = event.target.value;
    updateControllerMode(controller.cv_tag, newMode);
  });
  modeWrapper.appendChild(modeLabel);
  modeWrapper.appendChild(modeSelect);
  card.appendChild(modeWrapper);

  const pvRow = document.createElement("div");
  pvRow.className = "controller-card__row controller-card__row--status";
  const pvStatus = controller.pv.ok ? "OK" : "Fault";
  pvRow.innerHTML = `PV: <strong>${formatValue(controller.pv.value)}</strong> ${controller.pv.unit} <span class="status ${controller.pv.ok ? "status--ok" : "status--bad"}">${pvStatus}</span>`;
  card.appendChild(pvRow);

  const mvRow = document.createElement("div");
  mvRow.className = "controller-card__row controller-card__row--status";
  mvRow.innerHTML = `MV (${controller.manipulated.tag}): <strong>${formatValue(controller.manipulated.value)}</strong> ${controller.manipulated.unit}`;
  card.appendChild(mvRow);

  const form = document.createElement("form");
  form.className = "controller-card__setpoint";
  const setpointLabel = document.createElement("label");
  setpointLabel.textContent = `Setpoint (${controller.setpoint.unit})`;
  const setpointInput = document.createElement("input");
  setpointInput.type = "number";
  setpointInput.step = "0.01";
  setpointInput.value = controller.setpoint.value ?? "";
  const applyButton = document.createElement("button");
  applyButton.type = "submit";
  applyButton.textContent = "Apply";

  const inAuto = controller.mode === "AUTO";
  setpointInput.disabled = !inAuto;
  applyButton.disabled = !inAuto;
  if (!inAuto) {
    applyButton.title = "Setpoint changes available only in AUTO mode.";
  }

  form.appendChild(setpointLabel);
  form.appendChild(setpointInput);
  form.appendChild(applyButton);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const value = parseFloat(setpointInput.value);
    if (Number.isNaN(value)) {
      showMessage("Please enter a numeric setpoint value.", "error");
      return;
    }
    updateControllerSetpoint(controller.cv_tag, value);
  });
  card.appendChild(form);

  if (controller.children && controller.children.length) {
    const childContainer = document.createElement("div");
    childContainer.className = "controller-card__children";
    controller.children.forEach((child) => {
      childContainer.appendChild(buildControllerCard(child));
    });
    card.appendChild(childContainer);
  }

  return card;
}

function renderControllers(payload) {
  const container = document.getElementById("controllers");
  if (!container) return;
  container.innerHTML = "";
  const controllers = payload.controllers || [];
  if (!controllers.length) {
    container.innerHTML = "<p>No controllers configured.</p>";
    return;
  }
  controllers.forEach((controller) => {
    container.appendChild(buildControllerCard(controller));
  });
}

async function fetchControllers() {
  try {
    const response = await fetch("/api/controllers");
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    renderControllers(payload);
    updateTimeAndSpeed(payload.time, payload.speed);
  } catch (error) {
    console.error("Failed to load controller metadata", error);
    showMessage(`Unable to load controllers: ${error}`, "error");
  }
}

async function fetchTrends() {
  try {
    const response = await fetch("/api/trends?points=300");
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    updateCharts(payload);
    updateTimeAndSpeed(payload.time, payload.speed);
  } catch (error) {
    console.error("Failed to load trend data", error);
  }
}

async function updateControllerMode(cvTag, mode) {
  try {
    const response = await fetch(`/api/controllers/${encodeURIComponent(cvTag)}/mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    renderControllers(payload);
    updateTimeAndSpeed(payload.time, payload.speed);
    showMessage(`Updated ${cvTag} mode to ${mode}.`, "success");
  } catch (error) {
    console.error("Failed to update controller mode", error);
    showMessage(`Unable to change mode: ${error}`, "error");
    fetchControllers();
  }
}

async function updateControllerSetpoint(cvTag, value) {
  try {
    const response = await fetch(`/api/controllers/${encodeURIComponent(cvTag)}/setpoint`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ setpoint: value }),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    renderControllers(payload);
    updateTimeAndSpeed(payload.time, payload.speed);
    showMessage(`Setpoint for ${cvTag} updated.`, "success");
  } catch (error) {
    console.error("Failed to update setpoint", error);
    showMessage(`Unable to change setpoint: ${error}`, "error");
    fetchControllers();
  }
}

async function onSpeedSubmit(event) {
  event.preventDefault();
  const input = document.getElementById("speed-input");
  if (!input) return;
  const value = parseFloat(input.value);
  if (Number.isNaN(value) || value <= 0) {
    showMessage("Speed factor must be a positive number.", "error");
    return;
  }
  try {
    const response = await fetch("/api/speed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ factor: value }),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    document.getElementById("speed-factor").textContent = numberFormatter.format(payload.speed);
    showMessage(`Simulation speed set to ${numberFormatter.format(payload.speed)}×`, "success");
  } catch (error) {
    console.error("Failed to update speed", error);
    showMessage(`Unable to change speed: ${error}`, "error");
  }
}

function setupEventHandlers() {
  const speedForm = document.getElementById("speed-form");
  if (speedForm) {
    speedForm.addEventListener("submit", onSpeedSubmit);
  }
}

async function bootstrap() {
  initCharts();
  setupEventHandlers();
  await fetchControllers();
  await fetchTrends();
  setInterval(fetchTrends, 1500);
}

window.addEventListener("DOMContentLoaded", bootstrap);
