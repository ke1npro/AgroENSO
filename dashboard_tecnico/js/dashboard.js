/**
 * Lógica Interactiva del Dashboard Técnico AgroENSO
 * Soporta alternancia fluida entre:
 *   - Región Andina (window.DASHBOARD_DATA: Antioquia, Huila, Tolima)
 *   - Región Caribe (window.DASHBOARD_DATA_CARIBE: Atlántico, Bolívar, Cesar, Córdoba, La Guajira, Magdalena, Sucre)
 */

document.addEventListener("DOMContentLoaded", () => {
  let currentRegion = "andina";
  let activeCharts = [];

  function getRegionData() {
    if (currentRegion === "caribe") {
      return window.DASHBOARD_DATA_CARIBE || window.DASHBOARD_DATA;
    }
    return window.DASHBOARD_DATA;
  }

  // Paleta científica
  const colors = {
    blue: "#38BDF8",
    cyan: "#06B6D4",
    emerald: "#10B981",
    amber: "#F59E0B",
    red: "#EF4444",
    purple: "#8B5CF6",
    gray: "#475569",
    border: "#222F44",
    textMuted: "#94A3B8",
  };

  Chart.defaults.color = colors.textMuted;
  Chart.defaults.borderColor = "rgba(255, 255, 255, 0.06)";
  Chart.defaults.font.family = "'Plus Jakarta Sans', system-ui, sans-serif";

  // ---- 1. Control de Pestañas ----
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanels.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) targetPanel.classList.add("active");
    });
  });

  // ---- 2. Selector de Región (Andina vs Caribe) ----
  const regionBtns = document.querySelectorAll(".region-btn");
  regionBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      regionBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentRegion = btn.getAttribute("data-region");
      initDashboard();
    });
  });

  function destroyCharts() {
    activeCharts.forEach((c) => {
      try {
        c.destroy();
      } catch (e) {}
    });
    activeCharts = [];
  }

  // ---- 3. Función Principal de Renderizado del Dashboard ----
  function initDashboard() {
    destroyCharts();
    const data = getRegionData();
    if (!data) {
      console.error("No se encontraron datos para la región:", currentRegion);
      return;
    }

    const meta = data.metadata;

    // Actualizar píldoras superiores
    const pillReg = document.getElementById("pill-registros");
    if (pillReg) pillReg.innerHTML = `<span class="dot-live"></span> ${meta.registros_totales.toLocaleString()} Registros EVA`;
    const pillMun = document.getElementById("pill-municipios");
    if (pillMun) pillMun.textContent = `${meta.municipios} Municipios`;
    const pillDep = document.getElementById("pill-departamentos");
    if (pillDep) pillDep.textContent = meta.alcance;
    const pillPer = document.getElementById("pill-periodo");
    if (pillPer) pillPer.textContent = meta.periodo;

    // Actualizar KPI cards
    const kpiReg = document.getElementById("kpi-registros");
    if (kpiReg) kpiReg.textContent = meta.registros_totales.toLocaleString();
    const kpiMun = document.getElementById("kpi-municipios");
    if (kpiMun) kpiMun.textContent = meta.municipios;
    const kpiMunSub = document.getElementById("kpi-municipios-sub");
    if (kpiMunSub) {
      kpiMunSub.textContent = currentRegion === "caribe" 
        ? "Cesar · Magdalena · Sucre · Bolívar · Córdoba · La Guajira · Atlántico"
        : "124 Antioquia · 47 Tolima · 37 Huila";
    }
    const kpiPanel = document.getElementById("kpi-panel");
    if (kpiPanel) {
      kpiPanel.textContent = currentRegion === "caribe" ? "7,141" : "13,545";
    }
    const kpiPanelSub = document.getElementById("kpi-panel-sub");
    if (kpiPanelSub) {
      kpiPanelSub.textContent = currentRegion === "caribe"
        ? "Municipios Caribe con 18 años continuos"
        : "191 municipios con 18 años continuos";
    }

    // ==============================================================
    // PESTAÑA 1: EXPLORADOR DE DATOS Y CLIMA
    // ==============================================================

    // 1A. Gráfico del Índice ONI (1950 - 2025)
    const ctxOni = document.getElementById("chart-oni")?.getContext("2d");
    if (ctxOni && data.serie_oni) {
      const oniFiltered = data.serie_oni.filter((d) => d.año >= 1990);
      const chOni = new Chart(ctxOni, {
        type: "bar",
        data: {
          labels: oniFiltered.map((d) => d.año),
          datasets: [
            {
              label: "ONI Media Anual (°C)",
              data: oniFiltered.map((d) => d.oni_media),
              backgroundColor: oniFiltered.map((d) =>
                d.oni_media >= 0.5 ? "rgba(239, 68, 68, 0.75)" : (d.oni_media <= -0.5 ? "rgba(56, 189, 248, 0.75)" : "rgba(148, 163, 184, 0.35)")
              ),
              borderColor: oniFiltered.map((d) =>
                d.oni_media >= 0.5 ? colors.red : (d.oni_media <= -0.5 ? colors.blue : colors.gray)
              ),
              borderWidth: 1,
              borderRadius: 3,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => `ONI: ${ctx.raw > 0 ? "+" : ""}${ctx.raw}°C (${oniFiltered[ctx.dataIndex].categoria})`,
              },
            },
          },
          scales: {
            y: {
              title: { display: true, text: "Anomalía (°C)" },
              grid: { color: "rgba(255, 255, 255, 0.08)" },
            },
          },
        },
      });
      activeCharts.push(chOni);
    }

    // 1B. Rendimiento y Eficiencia por Cultivo
    const ctxCultivos = document.getElementById("chart-rendimiento-cultivos")?.getContext("2d");
    if (ctxCultivos && data.stats_cultivo) {
      const chCult = new Chart(ctxCultivos, {
        type: "bar",
        data: {
          labels: data.stats_cultivo.map((d) => d.cultivo),
          datasets: [
            {
              label: "Rendimiento Medio (t/ha)",
              data: data.stats_cultivo.map((d) => d.rend_medio),
              backgroundColor: "rgba(56, 189, 248, 0.65)",
              borderColor: colors.blue,
              borderWidth: 1,
              borderRadius: 4,
            },
            {
              label: "Eficiencia Media (clip)",
              data: data.stats_cultivo.map((d) => d.efic_clip_media),
              backgroundColor: "rgba(16, 185, 129, 0.65)",
              borderColor: colors.emerald,
              borderWidth: 1,
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: "top" } },
          scales: {
            y: { grid: { color: "rgba(255, 255, 255, 0.08)" } },
          },
        },
      });
      activeCharts.push(chCult);
    }

    // 1C. Clima Departamental vs ONI
    const ctxClima = document.getElementById("chart-clima-depto")?.getContext("2d");
    const selectDeptoClima = document.getElementById("select-depto-clima");
    let chartClimaInstance = null;

    function renderClimaDepto(depto) {
      if (!ctxClima || !data.clima_deptos) return;
      const filtered = data.clima_deptos.filter((d) => d.departamento === depto).sort((a, b) => a.año - b.año);

      if (chartClimaInstance) {
        chartClimaInstance.destroy();
        activeCharts = activeCharts.filter((c) => c !== chartClimaInstance);
      }
      chartClimaInstance = new Chart(ctxClima, {
        type: "line",
        data: {
          labels: filtered.map((d) => d.año),
          datasets: [
            {
              label: "Precipitación Anual (mm)",
              data: filtered.map((d) => d.precip_anual_mm),
              borderColor: colors.cyan,
              backgroundColor: "rgba(6, 182, 212, 0.1)",
              yAxisID: "yPrecip",
              tension: 0.3,
              fill: true,
            },
            {
              label: "Temperatura Media (°C)",
              data: filtered.map((d) => d.temp_media_c),
              borderColor: colors.amber,
              backgroundColor: "transparent",
              yAxisID: "yTemp",
              borderDash: [5, 5],
              tension: 0.3,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            yPrecip: {
              type: "linear",
              position: "left",
              title: { display: true, text: "Precipitación (mm)" },
              grid: { color: "rgba(255, 255, 255, 0.06)" },
            },
            yTemp: {
              type: "linear",
              position: "right",
              title: { display: true, text: "Temp (°C)" },
              grid: { drawOnChartArea: false },
            },
          },
        },
      });
      activeCharts.push(chartClimaInstance);
    }

    if (selectDeptoClima && data.clima_deptos) {
      const deptos = [...new Set(data.clima_deptos.map((d) => d.departamento))].sort();
      selectDeptoClima.innerHTML = deptos.map((d) => `<option value="${d}">${d}</option>`).join("");
      renderClimaDepto(deptos[0]);
      selectDeptoClima.onchange = (e) => renderClimaDepto(e.target.value);
    }

    // Llenar tabla de estadísticas por cultivo
    const tbodyStats = document.getElementById("tbody-stats-cultivos");
    if (tbodyStats && data.stats_cultivo) {
      tbodyStats.innerHTML = data.stats_cultivo
        .map(
          (c) => `
          <tr>
            <td><strong>${c.cultivo}</strong></td>
            <td>${c.registros.toLocaleString()}</td>
            <td>${c.rend_medio.toFixed(3)} t/ha</td>
            <td>${c.rend_mediana.toFixed(3)} t/ha</td>
            <td><strong>${c.efic_media.toFixed(3)}</strong></td>
            <td>${c.efic_clip_media.toFixed(3)}</td>
            <td>${c.efic_std.toFixed(3)}</td>
            <td>${c.area_media_ha.toFixed(1)} ha</td>
            <td>${c.prod_media_t.toFixed(1)} t</td>
          </tr>
        `
        )
        .join("");
    }

    // ==============================================================
    // PESTAÑA 2: BENCHMARKING DE BASELINES (LOEO)
    // ==============================================================
    const ctxBaseErr = document.getElementById("chart-baselines-error")?.getContext("2d");
    const ctxBaseR2 = document.getElementById("chart-baselines-r2")?.getContext("2d");
    let chartBaseErrInstance = null;
    let chartBaseR2Instance = null;

    function renderBaselines(splitKey) {
      let rows = [];
      if (splitKey === "promedio") {
        rows = data.promedio_loeo.filter((r) => r.configuracion_enso === "Con_ENSO");
      } else {
        rows = data.tabla_baselines.filter((r) => r.split === splitKey && r.configuracion_enso === "Con_ENSO");
      }

      const modelos = ["Ridge", "Huber", "RandomForest", "XGBoost", "XGBoost_Combinado", "RedModular_Sigmoide", "RedModular_FiLM"];
      const getVal = (m, field) => {
        const item = rows.find((r) => r.modelo === m);
        return item ? item[field] : 0;
      };

      const rmseVals = modelos.map((m) => getVal(m, "RMSE_test"));
      const maeVals = modelos.map((m) => getVal(m, "MAE_test"));
      const r2Vals = modelos.map((m) => getVal(m, "R2_test"));
      const rCorrVals = modelos.map((m) => getVal(m, "Pearson_r_test"));

      // 1. Gráfico Error (RMSE y MAE)
      if (ctxBaseErr) {
        if (chartBaseErrInstance) {
          chartBaseErrInstance.destroy();
          activeCharts = activeCharts.filter((c) => c !== chartBaseErrInstance);
        }
        chartBaseErrInstance = new Chart(ctxBaseErr, {
          type: "bar",
          data: {
            labels: modelos,
            datasets: [
              {
                label: "RMSE Test",
                data: rmseVals,
                backgroundColor: "rgba(239, 68, 68, 0.7)",
                borderColor: colors.red,
                borderWidth: 1,
                borderRadius: 4,
              },
              {
                label: "MAE Test",
                data: maeVals,
                backgroundColor: "rgba(245, 158, 11, 0.7)",
                borderColor: colors.amber,
                borderWidth: 1,
                borderRadius: 4,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              title: { display: true, text: "Errores de Generalización (Menor es Mejor)" },
            },
            scales: {
              y: { grid: { color: "rgba(255, 255, 255, 0.08)" } },
            },
          },
        });
        activeCharts.push(chartBaseErrInstance);
      }

      // 2. Gráfico Determinación (R2 y Pearson r)
      if (ctxBaseR2) {
        if (chartBaseR2Instance) {
          chartBaseR2Instance.destroy();
          activeCharts = activeCharts.filter((c) => c !== chartBaseR2Instance);
        }
        chartBaseR2Instance = new Chart(ctxBaseR2, {
          type: "bar",
          data: {
            labels: modelos,
            datasets: [
              {
                label: "R² Test",
                data: r2Vals,
                backgroundColor: "rgba(16, 185, 129, 0.7)",
                borderColor: colors.emerald,
                borderWidth: 1,
                borderRadius: 4,
              },
              {
                label: "Pearson r",
                data: rCorrVals,
                backgroundColor: "rgba(56, 189, 248, 0.7)",
                borderColor: colors.blue,
                borderWidth: 1,
                borderRadius: 4,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              title: { display: true, text: "Correlación y Determinación (Mayor es Mejor)" },
            },
            scales: {
              y: { grid: { color: "rgba(255, 255, 255, 0.08)" } },
            },
          },
        });
        activeCharts.push(chartBaseR2Instance);
      }

      // 3. Llenar tabla baselines
      const tbody = document.querySelector("#tabla-metricas-baselines tbody");
      if (tbody) {
        let fullRows = [];
        if (splitKey === "promedio") {
          fullRows = data.promedio_loeo;
        } else {
          fullRows = data.tabla_baselines.filter((r) => r.split === splitKey);
        }

        tbody.innerHTML = fullRows
          .map((r) => {
            const badgeClass =
              r.modelo === "XGBoost" ? "badge-xgboost" : (r.modelo === "RandomForest" ? "badge-rf" : (r.modelo === "Huber" ? "badge-huber" : "badge-ridge"));
            return `
              <tr>
                <td><span class="badge-model ${badgeClass}">${r.modelo}</span></td>
                <td>${r.configuracion_enso === "Con_ENSO" ? "✅ Con ONI" : "❌ Sin ONI"}</td>
                <td><strong>${r.RMSE_test.toFixed(4)}</strong></td>
                <td>${r.MAE_test.toFixed(4)}</td>
                <td>${r.R2_test.toFixed(4)}</td>
                <td>${r.Pearson_r_test.toFixed(4)}</td>
                <td>${r.Bias_test ? r.Bias_test.toFixed(4) : "—"}</td>
                <td>${r.RMSE_test_raw ? r.RMSE_test_raw.toFixed(4) : "—"}</td>
                <td>${r.tiempo_seg ? r.tiempo_seg + "s" : "—"}</td>
              </tr>
            `;
          })
          .join("");
      }
    }

    const filterBtns = document.querySelectorAll(".btn-filter");
    filterBtns.forEach((btn) => {
      btn.onclick = () => {
        filterBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        renderBaselines(btn.getAttribute("data-split"));
      };
    });

    renderBaselines("Split_1_LOEO_2015_2016");

    // Error por Cultivo (Split 2015-2016)
    const ctxErrCult = document.getElementById("chart-error-cultivo")?.getContext("2d");
    if (ctxErrCult && data.error_por_cultivo) {
      const cultivosUnicos = [...new Set(data.error_por_cultivo.map((d) => d.cultivo))];
      const modelos = ["Ridge", "RandomForest", "XGBoost", "RedModular_FiLM"];
      const datasets = modelos.map((m, idx) => {
        const colorList = [colors.gray, colors.blue, colors.emerald, colors.purple];
        return {
          label: m,
          data: cultivosUnicos.map((c) => {
            const item = data.error_por_cultivo.find((d) => d.modelo === m && d.cultivo === c);
            return item ? item.MAE : 0;
          }),
          backgroundColor: colorList[idx],
          borderRadius: 4,
        };
      });

      const chErrC = new Chart(ctxErrCult, {
        type: "bar",
        data: { labels: cultivosUnicos, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { title: { display: true, text: "MAE Test" } },
          },
        },
      });
      activeCharts.push(chErrC);
    }

    // ==============================================================
    // PESTAÑA 3: CURVAS DE ENTRENAMIENTO Y ABLACIÓN
    // ==============================================================
    const ctxCurva = document.getElementById("chart-curva-aprendizaje")?.getContext("2d");
    if (ctxCurva && data.curvas_aprendizaje) {
      const chCurv = new Chart(ctxCurva, {
        type: "line",
        data: {
          labels: data.curvas_aprendizaje.map((d) => d.iteracion),
          datasets: [
            {
              label: "Train Loss (RMSE)",
              data: data.curvas_aprendizaje.map((d) => d.train_loss),
              borderColor: colors.cyan,
              backgroundColor: "transparent",
              tension: 0.2,
            },
            {
              label: "Test Loss (LOEO 2015-2016)",
              data: data.curvas_aprendizaje.map((d) => d.test_loss),
              borderColor: colors.red,
              backgroundColor: "transparent",
              borderDash: [5, 5],
              tension: 0.2,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: { display: true, text: "Convergencia de XGBoost (Árboles vs Pérdida)" },
          },
          scales: {
            x: { title: { display: true, text: "Número de Estimadores (Iteración)" } },
            y: { title: { display: true, text: "Pérdida RMSE" } },
          },
        },
      });
      activeCharts.push(chCurv);
    }

    // Ablación ENSO (Comparación Con vs Sin ONI)
    const ctxAblacion = document.getElementById("chart-ablacion-enso")?.getContext("2d");
    if (ctxAblacion && data.promedio_loeo) {
      const modelos = ["Ridge", "Huber", "RandomForest", "XGBoost", "XGBoost_Combinado", "RedModular_Sigmoide", "RedModular_FiLM"];
      const rmseSin = modelos.map((m) => {
        const it = data.promedio_loeo.find((r) => r.modelo === m && r.configuracion_enso === "Sin_ENSO");
        return it ? it.RMSE_test : 0;
      });
      const rmseCon = modelos.map((m) => {
        const it = data.promedio_loeo.find((r) => r.modelo === m && r.configuracion_enso === "Con_ENSO");
        return it ? it.RMSE_test : 0;
      });

      const chAbl = new Chart(ctxAblacion, {
        type: "bar",
        data: {
          labels: modelos,
          datasets: [
            {
              label: "Sin Señal ENSO (Solo Clima Local)",
              data: rmseSin,
              backgroundColor: "rgba(148, 163, 184, 0.6)",
              borderColor: colors.gray,
              borderWidth: 1,
              borderRadius: 4,
            },
            {
              label: "Con Señal ENSO (ONI NOAA)",
              data: rmseCon,
              backgroundColor: "rgba(56, 189, 248, 0.75)",
              borderColor: colors.blue,
              borderWidth: 1,
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: { display: true, text: "Ablación ENSO en Promedio LOEO (RMSE: Menor es Mejor)" },
          },
          scales: {
            y: { grid: { color: "rgba(255, 255, 255, 0.08)" } },
          },
        },
      });
      activeCharts.push(chAbl);
    }

    // Feature Importances
    const ctxImp = document.getElementById("chart-feature-importances")?.getContext("2d");
    const selectModImp = document.getElementById("select-modelo-importances");
    let chartImpInstance = null;

    function renderImportances(mod) {
      if (!ctxImp || !data.feature_importances) return;
      const items = data.feature_importances[mod] ? data.feature_importances[mod].slice(0, 15) : [];

      if (chartImpInstance) {
        chartImpInstance.destroy();
        activeCharts = activeCharts.filter((c) => c !== chartImpInstance);
      }
      chartImpInstance = new Chart(ctxImp, {
        type: "bar",
        data: {
          labels: items.map((d) => d.feature),
          datasets: [
            {
              label: `Importancia en ${mod}`,
              data: items.map((d) => d.importance),
              backgroundColor: mod === "XGBoost" ? "rgba(16, 185, 129, 0.7)" : "rgba(56, 189, 248, 0.7)",
              borderColor: mod === "XGBoost" ? colors.emerald : colors.blue,
              borderWidth: 1,
              borderRadius: 4,
            },
          ],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { title: { display: true, text: "Gini / Gain Relativo" } },
          },
        },
      });
      activeCharts.push(chartImpInstance);
    }

    if (selectModImp) {
      renderImportances(selectModImp.value);
      selectModImp.onchange = (e) => renderImportances(e.target.value);
    }

    // ==============================================================
    // PESTAÑA 4: DIAGNÓSTICO RESIDUAL Y ESTACIONES IDEAM
    // ==============================================================
    const ctxScatter = document.getElementById("chart-scatter-pred")?.getContext("2d");
    let chartScatterInstance = null;

    if (ctxScatter && data.sample_predicciones) {
      const pts = data.sample_predicciones.map((d) => ({ x: d.y_real_clip, y: d.y_pred }));
      const maxVal = Math.max(...pts.map((p) => Math.max(p.x, p.y))) * 1.05;

      chartScatterInstance = new Chart(ctxScatter, {
        type: "scatter",
        data: {
          datasets: [
            {
              label: "Observado vs Predicho (XGBoost)",
              data: pts,
              backgroundColor: "rgba(56, 189, 248, 0.6)",
              borderColor: colors.blue,
              pointRadius: 3,
            },
            {
              label: "Línea Ideal 1:1 (y = x)",
              data: [
                { x: 0, y: 0 },
                { x: maxVal, y: maxVal },
              ],
              type: "line",
              borderColor: colors.red,
              borderDash: [5, 5],
              pointRadius: 0,
              fill: false,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: "Eficiencia Observada (clip)" }, min: 0, max: maxVal },
            y: { title: { display: true, text: "Eficiencia Predicha" }, min: 0, max: maxVal },
          },
        },
      });
      activeCharts.push(chartScatterInstance);
    }

    // Histograma de Residuos
    const ctxHist = document.getElementById("chart-hist-residuos")?.getContext("2d");
    if (ctxHist && data.sample_predicciones) {
      const residuals = data.sample_predicciones.map((d) => d.residual);
      const binCount = 20;
      const minR = Math.min(...residuals);
      const maxR = Math.max(...residuals);
      const step = (maxR - minR) / binCount;
      const bins = Array(binCount).fill(0);
      const labels = [];

      for (let i = 0; i < binCount; i++) {
        labels.push((minR + i * step + step / 2).toFixed(2));
      }
      residuals.forEach((r) => {
        let b = Math.floor((r - minR) / step);
        if (b >= binCount) b = binCount - 1;
        bins[b]++;
      });

      const chH = new Chart(ctxHist, {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "Frecuencia de Residuos",
              data: bins,
              backgroundColor: "rgba(139, 92, 246, 0.7)",
              borderColor: colors.purple,
              borderWidth: 1,
              borderRadius: 3,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { title: { display: true, text: "Residual (Pred - Real)" } },
            y: { title: { display: true, text: "Frecuencia" } },
          },
        },
      });
      activeCharts.push(chH);
    }

    // Auditoría de Calidad de Estaciones
    const ctxCalidad = document.getElementById("chart-calidad-estaciones")?.getContext("2d");
    if (ctxCalidad && data.calidad_estaciones) {
      const chCal = new Chart(ctxCalidad, {
        type: "bar",
        data: {
          labels: data.calidad_estaciones.map((d) => d.etiqueta),
          datasets: [
            {
              label: "MAE del Modelo",
              data: data.calidad_estaciones.map((d) => d.MAE),
              backgroundColor: ["rgba(239, 68, 68, 0.75)", "rgba(16, 185, 129, 0.75)"],
              borderColor: [colors.red, colors.emerald],
              borderWidth: 1,
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { title: { display: true, text: "Error Absoluto Medio (MAE)" } },
          },
        },
      });
      activeCharts.push(chCal);
    }

    const tbodyCalidad = document.getElementById("tbody-calidad-estaciones");
    if (tbodyCalidad && data.calidad_estaciones) {
      tbodyCalidad.innerHTML = data.calidad_estaciones
        .map(
          (c) => `
          <tr>
            <td><strong>${c.etiqueta}</strong></td>
            <td>${c.N.toLocaleString()}</td>
            <td><strong>${c.MAE.toFixed(4)}</strong></td>
            <td>${c.RMSE.toFixed(4)}</td>
            <td>${c.mediana_dist_km.toFixed(1)} km</td>
          </tr>
        `
        )
        .join("");
    }
  }

  // Inicialización con región por defecto
  initDashboard();
});
