document.addEventListener('DOMContentLoaded', () => {
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const videoStream = document.getElementById('video-stream');
    const videoPlaceholder = document.getElementById('video-placeholder');
    const led = document.getElementById('system-led');
    const statusText = document.getElementById('system-status-text');

    // Stats Elements
    const statMc = document.getElementById('stat-motorcycles');
    const statPerson = document.getElementById('stat-persons');
    const statGaps = document.getElementById('stat-gaps');
    const statFps = document.getElementById('stat-fps');

    let statsInterval = null;

    // --- Chart.js Setup ---
    const ctx = document.getElementById('realtimeChart').getContext('2d');
    const MAX_DATA_POINTS = 60; // 30 seconds if updated every 500ms

    const chartConfig = {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'FPS',
                    borderColor: 'hsla(120, 100%, 60%, 1)',
                    backgroundColor: 'hsla(120, 100%, 60%, 0.1)',
                    data: [],
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: 'Motorcycles',
                    borderColor: 'hsla(210, 100%, 60%, 1)',
                    backgroundColor: 'hsla(210, 100%, 60%, 0.1)',
                    data: [],
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: 'Persons',
                    borderColor: 'hsla(0, 100%, 60%, 1)',
                    backgroundColor: 'hsla(0, 100%, 60%, 0.1)',
                    data: [],
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: 'Gaps',
                    borderColor: 'hsla(45, 100%, 55%, 1)',
                    backgroundColor: 'hsla(45, 100%, 55%, 0.1)',
                    data: [],
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    labels: { color: '#ccc' }
                }
            },
            scales: {
                x: {
                    display: false // hide x axis labels for cleaner look
                },
                y: {
                    grid: { color: 'hsla(0, 0%, 100%, 0.05)' },
                    ticks: { color: '#888' },
                    beginAtZero: true
                }
            }
        }
    };
    const realtimeChart = new Chart(ctx, chartConfig);

    function updateChart(fps, mc, person, gaps) {
        const now = new Date();
        const timeStr = now.getHours() + ':' + now.getMinutes() + ':' + now.getSeconds();
        
        realtimeChart.data.labels.push(timeStr);
        realtimeChart.data.datasets[0].data.push(fps);
        realtimeChart.data.datasets[1].data.push(mc);
        realtimeChart.data.datasets[2].data.push(person);
        realtimeChart.data.datasets[3].data.push(gaps);

        if (realtimeChart.data.labels.length > MAX_DATA_POINTS) {
            realtimeChart.data.labels.shift();
            realtimeChart.data.datasets.forEach(dataset => dataset.data.shift());
        }

        realtimeChart.update();
    }

    function resetChart() {
        realtimeChart.data.labels = [];
        realtimeChart.data.datasets.forEach(dataset => dataset.data = []);
        realtimeChart.update();
    }
    // --- End Chart.js Setup ---

    // Helper: Animate number change
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = end; // Ensure exact end value
            }
        };
        window.requestAnimationFrame(step);
    }

    async function fetchStats() {
        try {
            const response = await fetch('/api/stats');
            const data = await response.json();
            
            if (data.is_running) {
                statMc.textContent = data.motorcycles;
                statPerson.textContent = data.persons;
                statGaps.textContent = data.available_gaps;
                statFps.textContent = data.fps.toFixed(1);
                
                // Update chart
                updateChart(data.fps, data.motorcycles, data.persons, data.available_gaps);
            } else {
                // If it stopped unexpectedly
                if (!btnStart.disabled && led.classList.contains('active')) {
                    stopStreamUI();
                }
            }
        } catch (error) {
            console.error("Error fetching stats:", error);
        }
    }

    function startStreamUI() {
        btnStart.disabled = true;
        btnStop.disabled = false;
        
        videoPlaceholder.classList.add('hidden');
        videoStream.classList.remove('hidden');
        
        // Add timestamp to bypass browser cache
        videoStream.src = `/api/stream?t=${new Date().getTime()}`;

        led.classList.add('active');
        statusText.textContent = "System Live";

        if (!statsInterval) {
            statsInterval = setInterval(fetchStats, 500);
        }
    }

    function stopStreamUI() {
        btnStart.disabled = false;
        btnStop.disabled = true;

        videoStream.src = "";
        videoStream.classList.add('hidden');
        videoPlaceholder.classList.remove('hidden');

        led.classList.remove('active');
        statusText.textContent = "System Idle";

        if (statsInterval) {
            clearInterval(statsInterval);
            statsInterval = null;
        }

        statMc.textContent = "0";
        statPerson.textContent = "0";
        statGaps.textContent = "0";
        statFps.textContent = "0.0";
        
        // Reset chart
        resetChart();
    }

    btnStart.addEventListener('click', async () => {
        try {
            await fetch('/api/control/start', { method: 'POST' });
            startStreamUI();
        } catch (err) {
            alert("Failed to start stream");
        }
    });

    btnStop.addEventListener('click', async () => {
        try {
            await fetch('/api/control/stop', { method: 'POST' });
            stopStreamUI();
        } catch (err) {
            alert("Failed to stop stream");
        }
    });
});
