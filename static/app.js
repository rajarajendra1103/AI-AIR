/* ==========================================================================
   AI AIR QUALITY & WEATHER FORECASTING - APPLICATION LOGIC
   ========================================================================== */

let forecastChartInstance = null;
let historicalChartInstance = null;
let benchmarkChartInstance = null;

let appData = {
  cities: [],
  profiles: [],
  models: []
};

// -------------------------------------------------------------
// 1. Theme Management (Light High-Contrast / Dark High-Contrast)
// -------------------------------------------------------------
function toggleTheme() {
  const htmlEl = document.documentElement;
  const currentTheme = htmlEl.getAttribute('data-theme') || 'light';
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';
  htmlEl.setAttribute('data-theme', newTheme);
  
  const iconEl = document.getElementById('themeIcon');
  const textEl = document.getElementById('themeText');
  
  if (newTheme === 'dark') {
    iconEl.textContent = '🌙';
    textEl.textContent = 'Dark High-Contrast';
  } else {
    iconEl.textContent = '☀️';
    textEl.textContent = 'Light High-Contrast';
  }

  // Re-render active charts with updated theme text colors
  updateChartThemes();
}

function getChartColors() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {
    textColor: isDark ? '#ffffff' : '#000000',
    gridColor: isDark ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.12)',
    primaryLine: isDark ? '#38bdf8' : '#0284c7',
    primaryFill: isDark ? 'rgba(56, 189, 248, 0.15)' : 'rgba(2, 132, 199, 0.12)'
  };
}

function updateChartThemes() {
  if (forecastChartInstance) forecastChartInstance.update();
  if (historicalChartInstance) historicalChartInstance.update();
  if (benchmarkChartInstance) benchmarkChartInstance.update();
}

// -------------------------------------------------------------
// 2. Tab Switcher
// -------------------------------------------------------------
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

  const targetContent = document.getElementById(`tab-${tabId}`);
  if (targetContent) targetContent.classList.add('active');

  // Activate matching button
  const matchingBtn = Array.from(document.querySelectorAll('.tab-btn')).find(btn => 
    btn.getAttribute('onclick').includes(tabId)
  );
  if (matchingBtn) matchingBtn.classList.add('active');

  if (tabId === 'historical' && (!historicalChartInstance || document.getElementById('histCitySelect').options.length <= 1)) {
    loadHistoricalData();
  } else if (tabId === 'benchmarks' && !benchmarkChartInstance) {
    loadBenchmarks();
  }
}

// -------------------------------------------------------------
// 3. Application Initialization
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch('/api/cities');
    const data = await res.json();

    appData.cities = data.cities || [];
    appData.profiles = data.profiles || [];
    appData.models = data.models || [];

    // Populate profile select
    const profileSelect = document.getElementById('profileSelect');
    profileSelect.innerHTML = appData.profiles.map(p => 
      `<option value="${p}" ${p.includes('Asthma') ? 'selected' : ''}>${p}</option>`
    ).join('');

    // Populate historical city select
    const histCitySelect = document.getElementById('histCitySelect');
    histCitySelect.innerHTML = appData.cities.map(c => 
      `<option value="${c}" ${c === 'Delhi' ? 'selected' : ''}>${c}</option>`
    ).join('');

    // Initial Live Forecast Load
    loadLiveForecast();

  } catch (err) {
    console.error("Initialization error:", err);
  }
});

// -------------------------------------------------------------
// 4. Live Open-Meteo & AI Forecast
// -------------------------------------------------------------
async function loadLiveForecast() {
  const city = document.getElementById('cityInput').value.trim() || 'Delhi';
  const modelName = document.getElementById('modelSelect').value;
  const profile = document.getElementById('profileSelect').value;

  const loader = document.getElementById('forecastLoader');
  const resultsContainer = document.getElementById('forecastResults');

  loader.style.display = 'flex';
  resultsContainer.style.display = 'none';

  try {
    const res = await fetch(`/api/forecast?city=${encodeURIComponent(city)}&model_name=${encodeURIComponent(modelName)}&profile=${encodeURIComponent(profile)}`);
    if (!res.ok) {
      const err = await res.json();
      alert(`Error: ${err.detail || 'Failed to fetch forecast.'}`);
      loader.style.display = 'none';
      return;
    }

    const data = await res.json();
    renderLiveForecast(data);

    loader.style.display = 'none';
    resultsContainer.style.display = 'block';

  } catch (err) {
    console.error("Forecast fetch error:", err);
    alert("Unable to fetch telemetry or prediction. Please check network connection.");
    loader.style.display = 'none';
  }
}

function renderLiveForecast(data) {
  const loc = data.location;
  const tel = data.telemetry;
  const aq = tel.air_quality;
  const w = tel.weather;
  const fc = data.forecast;
  const adv = data.health_advisory;

  // Location badge
  document.getElementById('locationBadge').innerHTML = `
    📍 Location Resolved: <strong>${loc.name}, ${loc.country}</strong> 
    (Lat: ${loc.lat.toFixed(2)}, Lon: ${loc.lon.toFixed(2)})
  `;

  // Weather alert
  const alertEl = document.getElementById('weatherAlert');
  if (data.weather_alert) {
    alertEl.innerHTML = `⚠️ <strong>Weather Impact Alert</strong>: ${data.weather_alert}`;
    alertEl.style.display = 'flex';
  } else {
    alertEl.style.display = 'none';
  }

  // Telemetry metrics
  document.getElementById('tempVal').textContent = w.temperature.toFixed(1);
  document.getElementById('humidityVal').textContent = w.humidity;
  document.getElementById('windVal').textContent = w.wind_speed.toFixed(1);
  document.getElementById('pressureVal').textContent = w.pressure.toFixed(0);

  const aqiValEl = document.getElementById('aqiVal');
  aqiValEl.textContent = aq.AQI.toFixed(0);
  
  const categoryEl = document.getElementById('aqiCategory');
  categoryEl.textContent = adv.AQI_Category;

  document.getElementById('majorPollutant').textContent = aq.Major_Pollutant || 'PM2.5';

  // Pollutants
  document.getElementById('pm25Val').textContent = aq['PM2.5'].toFixed(1);
  document.getElementById('pm10Val').textContent = aq['PM10'].toFixed(1);
  document.getElementById('no2Val').textContent = aq['NO2'].toFixed(1);
  document.getElementById('so2Val').textContent = aq['SO2'].toFixed(1);
  document.getElementById('coVal').textContent = aq['CO'].toFixed(2);
  document.getElementById('o3Val').textContent = aq['O3'].toFixed(1);

  // Predictions
  document.getElementById('pred1d').textContent = fc.pred_1d_aqi.toFixed(1);
  document.getElementById('pred3d').textContent = fc.pred_3d_aqi.toFixed(1);
  document.getElementById('pred7d').textContent = fc.pred_7d_aqi.toFixed(1);

  // Advisory details
  document.getElementById('riskLevelVal').textContent = adv.Health_Risk_Level;
  document.getElementById('safetyScoreVal').textContent = adv.Personalized_Safety_Score;
  document.getElementById('targetProfileVal').textContent = adv.Profile;
  document.getElementById('recommendedActionVal').textContent = adv.Recommended_Action;
  document.getElementById('maskGuidanceVal').textContent = adv.Mask_Guidance;
  document.getElementById('purifierGuidanceVal').textContent = adv.Air_Purifier_Guidance;
  
  const warnings = adv.Pollutant_Warnings && adv.Pollutant_Warnings.length > 0
    ? adv.Pollutant_Warnings.join(" | ")
    : `Primary Pollutant: ${aq.Major_Pollutant}`;
  document.getElementById('threatFactorsVal').textContent = warnings;

  // Render Forecast Chart
  renderForecastChart(fc.timeline, fc.model_used);
}

function renderForecastChart(timeline, modelName) {
  const ctx = document.getElementById('forecastChart').getContext('2d');
  const labels = timeline.map(t => t.date);
  const values = timeline.map(t => t.aqi);

  const colors = getChartColors();

  if (forecastChartInstance) {
    forecastChartInstance.destroy();
  }

  forecastChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: `PyTorch ${modelName} 7-Day Forecast (AQI)`,
        data: values,
        borderColor: colors.primaryLine,
        backgroundColor: colors.primaryFill,
        borderWidth: 3,
        pointBackgroundColor: colors.primaryLine,
        pointRadius: 5,
        pointHoverRadius: 7,
        fill: true,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: colors.textColor,
            font: { family: 'Plus Jakarta Sans', size: 14, weight: '700' }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: colors.textColor, font: { weight: '600' } },
          grid: { color: colors.gridColor }
        },
        y: {
          ticks: { color: colors.textColor, font: { weight: '600' } },
          grid: { color: colors.gridColor },
          title: { display: true, text: 'CPCB Air Quality Index (AQI)', color: colors.textColor, font: { weight: '700' } }
        }
      }
    }
  });
}

// -------------------------------------------------------------
// 5. Historical Data Explorer
// -------------------------------------------------------------
async function loadHistoricalData() {
  const city = document.getElementById('histCitySelect').value || 'Delhi';
  const loader = document.getElementById('histLoader');
  const results = document.getElementById('histResults');

  loader.style.display = 'flex';
  results.style.display = 'none';

  try {
    const res = await fetch(`/api/historical?city=${encodeURIComponent(city)}`);
    if (!res.ok) {
      alert("Failed to load historical data.");
      loader.style.display = 'none';
      return;
    }
    const data = await res.json();

    document.getElementById('histAvgAqi').textContent = data.summary.avg_aqi;
    document.getElementById('histMaxAqi').textContent = data.summary.max_aqi;
    document.getElementById('histAvgPm25').textContent = data.summary.avg_pm25;
    document.getElementById('histAvgPm10').textContent = data.summary.avg_pm10;
    document.getElementById('histAvgNo2').textContent = data.summary.avg_no2;

    renderHistoricalChart(data.records, city);

    loader.style.display = 'none';
    results.style.display = 'block';

  } catch (err) {
    console.error("Historical data error:", err);
    loader.style.display = 'none';
  }
}

function renderHistoricalChart(records, city) {
  const ctx = document.getElementById('historicalChart').getContext('2d');
  const labels = records.map(r => r.Date);
  const aqis = records.map(r => r.AQI);
  const colors = getChartColors();

  if (historicalChartInstance) {
    historicalChartInstance.destroy();
  }

  historicalChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: `Historical Daily AQI Trend for ${city}`,
        data: aqis,
        borderColor: '#e11d48',
        backgroundColor: 'rgba(225, 29, 72, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 5,
        fill: true,
        tension: 0.1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: colors.textColor, font: { weight: '700', size: 14 } }
        }
      },
      scales: {
        x: {
          ticks: { color: colors.textColor, maxTicksLimit: 12, font: { weight: '600' } },
          grid: { color: colors.gridColor }
        },
        y: {
          ticks: { color: colors.textColor, font: { weight: '600' } },
          grid: { color: colors.gridColor }
        }
      }
    }
  });
}

// -------------------------------------------------------------
// 6. PyTorch Model Benchmarks
// -------------------------------------------------------------
async function loadBenchmarks() {
  try {
    const res = await fetch('/api/benchmarks');
    const data = await res.json();
    const stats = data.benchmarks || {};

    const tbody = document.querySelector('#benchmarkTable tbody');
    tbody.innerHTML = '';

    const models = Object.keys(stats);
    const maes = [];
    const rmses = [];

    models.forEach(m => {
      const s = stats[m];
      maes.push(s.MAE);
      rmses.push(s.RMSE);

      tbody.innerHTML += `
        <tr>
          <td><strong>${m} Forecaster</strong></td>
          <td>${s.MAE.toFixed(3)}</td>
          <td>${s.RMSE.toFixed(3)}</td>
          <td>${s.MAPE.toFixed(2)}%</td>
          <td><strong>${s.R2.toFixed(4)}</strong></td>
        </tr>
      `;
    });

    renderBenchmarkChart(models, maes, rmses);

  } catch (err) {
    console.error("Benchmarks error:", err);
  }
}

function renderBenchmarkChart(models, maes, rmses) {
  const ctx = document.getElementById('benchmarkChart').getContext('2d');
  const colors = getChartColors();

  if (benchmarkChartInstance) {
    benchmarkChartInstance.destroy();
  }

  benchmarkChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: models,
      datasets: [
        {
          label: 'MAE (Lower is Better)',
          data: maes,
          backgroundColor: '#0284c7'
        },
        {
          label: 'RMSE (Lower is Better)',
          data: rmses,
          backgroundColor: '#7c3aed'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: colors.textColor, font: { weight: '700', size: 14 } }
        }
      },
      scales: {
        x: {
          ticks: { color: colors.textColor, font: { weight: '700', size: 13 } },
          grid: { color: colors.gridColor }
        },
        y: {
          ticks: { color: colors.textColor, font: { weight: '600' } },
          grid: { color: colors.gridColor }
        }
      }
    }
  });
}
