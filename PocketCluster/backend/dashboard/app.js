/* ──────────────────────────────────────────────────────────────
   PocketCluster Admin Dashboard — Client Logic
   Fetches /admin/dashboard and renders all sections.
   Auto-refreshes every 10 seconds.
   ────────────────────────────────────────────────────────────── */

const API_URL = 'http://localhost:8000/admin/dashboard';
const REFRESH_MS = 10_000;

// ── Helpers ─────────────────────────────────────────────────

function formatBytes(bytes) {
    if (bytes == null || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(str) {
    if (!str || str === 'None') return '—';
    const d = new Date(str);
    if (isNaN(d)) return str;
    return d.toLocaleDateString('en-IN', {
        day: 'numeric', month: 'short', year: 'numeric',
    }) + ' ' + d.toLocaleTimeString('en-IN', {
        hour: '2-digit', minute: '2-digit',
    });
}

function timeAgo(str) {
    if (!str || str === 'None') return '—';
    const d = new Date(str);
    if (isNaN(d)) return str;
    const diff = (Date.now() - d.getTime()) / 1000;
    if (diff < 60) return `${Math.floor(diff)}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

// Animated number counter
function animateValue(el, start, end, duration = 600) {
    if (start === end) { el.textContent = end; return; }
    const range = end - start;
    const startTime = performance.now();
    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        el.textContent = Math.floor(start + range * ease);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ── Set status indicator ────────────────────────────────────

function setConnectionStatus(status, text) {
    const dot = document.querySelector('#connection-status .pulse');
    const label = document.querySelector('#connection-status .header__status-text');
    dot.className = 'pulse ' + status; // 'online', 'error', ''
    label.textContent = text;
}

// ── Render Functions ────────────────────────────────────────

function renderOverview(ov) {
    const map = {
        'card-users': ov.total_users,
        'card-devices': ov.total_devices,
        'card-online': ov.online_devices,
        'card-offline': ov.offline_devices,
        'card-files': ov.total_files,
        'card-chunks': ov.total_chunks,
    };
    for (const [id, val] of Object.entries(map)) {
        const el = document.querySelector(`#${id} .card__value`);
        const prev = parseInt(el.dataset.target) || 0;
        el.dataset.target = val;
        animateValue(el, prev, val);
    }

    // Storage bar
    const used = ov.total_storage_used || 0;
    const cap = ov.total_capacity || 1;
    const pct = Math.min((used / cap) * 100, 100);
    document.getElementById('storage-text').textContent =
        `${formatBytes(used)} / ${formatBytes(cap)}`;
    document.getElementById('storage-fill').style.width = pct + '%';
}

function renderDevices(devices) {
    const tbody = document.getElementById('devices-tbody');
    const empty = document.getElementById('devices-empty');
    document.getElementById('device-count-badge').textContent = devices.length;

    if (devices.length === 0) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';

    tbody.innerHTML = devices.map(d => {
        const isOn = d.status === 'ONLINE';
        const statusClass = isOn ? 'status--online' : 'status--offline';
        const modeClass = d.mode === 'Cluster' ? 'mode-badge--cluster' : 'mode-badge--user';

        const cap = d.storage_capacity || 1;
        const used = cap - (d.available_storage || 0);
        const pct = Math.min((used / cap) * 100, 100);

        return `<tr>
            <td>${d.device_id}</td>
            <td>${d.device_name || '—'}</td>
            <td><span class="status ${statusClass}"><span class="status__dot"></span>${d.status}</span></td>
            <td><span class="mode-badge ${modeClass}">${d.mode}</span></td>
            <td>
                <div class="mini-storage">
                    <div class="mini-storage__bar"><div class="mini-storage__fill" style="width:${pct}%"></div></div>
                    ${formatBytes(used)} / ${formatBytes(cap)}
                </div>
            </td>
            <td>${timeAgo(d.last_heartbeat)}</td>
        </tr>`;
    }).join('');
}

function renderFiles(files) {
    const tbody = document.getElementById('files-tbody');
    const empty = document.getElementById('files-empty');
    document.getElementById('file-count-badge').textContent = files.length;

    if (files.length === 0) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';

    tbody.innerHTML = files.map(f => `<tr>
        <td>${f.file_id}</td>
        <td>${f.file_name}</td>
        <td>${f.owner_email}</td>
        <td>${formatBytes(f.file_size)}</td>
        <td>${f.num_chunks}</td>
        <td>${formatDate(f.upload_timestamp)}</td>
    </tr>`).join('');
}

function renderUsers(users) {
    const tbody = document.getElementById('users-tbody');
    const empty = document.getElementById('users-empty');
    document.getElementById('user-count-badge').textContent = users.length;

    if (users.length === 0) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';

    tbody.innerHTML = users.map(u => `<tr>
        <td>${u.user_id}</td>
        <td>${u.email}</td>
        <td>${u.file_count}</td>
        <td>${formatDate(u.created_at)}</td>
        <td>${u.last_login && u.last_login !== 'None' ? formatDate(u.last_login) : '—'}</td>
    </tr>`).join('');
}

function renderReplication(summary) {
    const active = summary['ACTIVE'] || 0;
    const replicating = summary['REPLICATING'] || 0;
    const lost = summary['LOST'] || 0;
    const failed = summary['FAILED'] || 0;
    const total = active + replicating + lost + failed || 1;

    document.querySelector('#repl-active .repl-card__value').textContent = active;
    document.querySelector('#repl-replicating .repl-card__value').textContent = replicating;
    document.querySelector('#repl-lost .repl-card__value').textContent = lost;
    document.querySelector('#repl-failed .repl-card__value').textContent = failed;

    // Stacked bar
    const bar = document.getElementById('repl-bar');
    bar.innerHTML = [
        { cls: 'active', val: active },
        { cls: 'replicating', val: replicating },
        { cls: 'lost', val: lost },
        { cls: 'failed', val: failed },
    ].filter(s => s.val > 0)
        .map(s => `<div class="repl-bar__seg repl-bar__seg--${s.cls}" style="width:${(s.val / total * 100).toFixed(1)}%" title="${s.cls}: ${s.val}"></div>`)
        .join('');
}

// ── Chart.js Setup ────────────────────────────────────────────

Chart.defaults.color = '#7a8baa';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.scale.grid.color = 'rgba(255, 255, 255, 0.05)';

let clusterStorageChart, deviceStatusChart, replicationChart, uploadActivityChart, nodeStorageChart;

function initCharts() {
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } }
    };

    // 1. Cluster Storage (Doughnut)
    clusterStorageChart = new Chart(document.getElementById('clusterStorageChart'), {
        type: 'doughnut',
        data: { labels: ['Used', 'Free'], datasets: [{ data: [0, 1], backgroundColor: ['#00e5c7', 'rgba(255,255,255,0.06)'], borderWidth: 0 }] },
        options: { ...commonOptions, cutout: '75%' }
    });

    // 2. Device Status (Bar)
    deviceStatusChart = new Chart(document.getElementById('deviceStatusChart'), {
        type: 'bar',
        data: { labels: ['Online', 'Offline'], datasets: [{ label: 'Devices', data: [0, 0], backgroundColor: ['#22c55e', '#ef4444'], borderRadius: 6 }] },
        options: { ...commonOptions, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } } }
    });

    // 3. Replication Health (Pie)
    replicationChart = new Chart(document.getElementById('replicationChart'), {
        type: 'pie',
        data: { labels: ['Active', 'Replicating', 'Lost', 'Failed'], datasets: [{ data: [0, 0, 0, 0], backgroundColor: ['#22c55e', '#38bdf8', '#facc15', '#ef4444'], borderWidth: 0 }] },
        options: commonOptions
    });

    // 4. Upload Activity (Line)
    uploadActivityChart = new Chart(document.getElementById('uploadActivityChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Files Uploaded', data: [], borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.1)', fill: true, tension: 0.3, pointRadius: 4 }] },
        options: { ...commonOptions, scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } } }
    });

    // 5. Node Storage (Bar)
    nodeStorageChart = new Chart(document.getElementById('nodeStorageChart'), {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'Used (%)', data: [], backgroundColor: '#a78bfa', borderRadius: 4 }] },
        options: { ...commonOptions, scales: { y: { beginAtZero: true, max: 100 } } }
    });
}

function updateCharts(data) {
    // 1. Storage
    const used = data.overview.total_storage_used || 0;
    const cap = data.overview.total_capacity || 1;
    clusterStorageChart.data.datasets[0].data = [used, cap - used];
    clusterStorageChart.update();

    // 2. Device Status
    deviceStatusChart.data.datasets[0].data = [data.overview.online_devices, data.overview.offline_devices];
    deviceStatusChart.update();

    // 3. Replication
    const rs = data.replica_summary;
    replicationChart.data.datasets[0].data = [rs['ACTIVE'] || 0, rs['REPLICATING'] || 0, rs['LOST'] || 0, rs['FAILED'] || 0];
    replicationChart.update();

    // 4. Upload Activity (group by day)
    const dates = {};
    data.files.forEach(f => {
        if (!f.upload_timestamp) return;
        const d = new Date(f.upload_timestamp).toLocaleDateString();
        dates[d] = (dates[d] || 0) + 1;
    });
    // Sort chronologically
    const sortedDates = Object.keys(dates).sort((a,b) => new Date(a) - new Date(b));
    uploadActivityChart.data.labels = sortedDates;
    uploadActivityChart.data.datasets[0].data = sortedDates.map(d => dates[d]);
    uploadActivityChart.update();

    // 5. Node Storage (Cluster mode only)
    const clusterNodes = data.devices.filter(d => d.mode === 'Cluster');
    nodeStorageChart.data.labels = clusterNodes.map(d => d.device_name || d.device_id);
    nodeStorageChart.data.datasets[0].data = clusterNodes.map(d => {
        const c = d.storage_capacity || 1;
        const u = c - (d.available_storage || 0);
        return Math.min((u / c) * 100, 100).toFixed(1);
    });
    nodeStorageChart.update();
}

// ── Main fetch + render ─────────────────────────────────────

async function fetchDashboard() {
    try {
        const res = await fetch(API_URL);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // calculate real storage usage from devices
        let totalCapacity = 0;
        let totalUsed = 0;

        data.devices.forEach(d => {
            if (d.status !== "ONLINE") return;

            const cap = d.storage_capacity || 0;
            const avail = d.available_storage || 0;
            const used = cap - avail;

            totalCapacity += cap;
            totalUsed += used;
        });

        data.overview.total_capacity = totalCapacity;
        data.overview.total_storage_used = totalUsed;
        data.overview.total_available = totalCapacity - totalUsed;

        renderOverview(data.overview);
        renderDevices(data.devices);
        renderFiles(data.files);
        renderUsers(data.users);
        renderReplication(data.replica_summary);

        updateCharts(data);
        
        
        renderOverview(data.overview);
        renderDevices(data.devices);
        renderFiles(data.files);
        renderUsers(data.users);
        renderReplication(data.replica_summary);
        
        updateCharts(data);

        setConnectionStatus('online', 'Live');
        document.getElementById('last-refresh').textContent =
            'Last refresh: ' + new Date().toLocaleTimeString();
    } catch (err) {
        console.error('Dashboard fetch error:', err);
        setConnectionStatus('error', 'Disconnected');
    }
}

// Initial load
initCharts();
fetchDashboard();

// Auto-refresh
setInterval(fetchDashboard, REFRESH_MS);
