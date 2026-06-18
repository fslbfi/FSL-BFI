var _chartInstance = null;

function app() {
    return {
        connected: false,
        mode: 'replay',
        current: {
            prediction: null, confidence: null, p_empty: null, p_occupied: null,
            dist_empty: null, dist_occupied: null, ground_truth: null,
            replay_file: null, device: null
        },
        totalWindows: 0,
        logEntries: [],
        logId: 0,

        allPredictions: [],
        filteredPredictions: [],
        displayData: [],
        totalTime: 0,
        currentTime: 0,
        currentIdx: 0,

        selectedFile: 'all',
        availableFiles: [],

        playing: false,
        speed: 1,
        _animFrame: null,
        _lastFrameTime: null,

        showVideo: false,
        videoSrc: '',
        videoOffsetMs: 0,

        liveInterface: 'wlan0mon',
        liveDeviceName: 'M7',
        liveDeviceStandard: 'AC',
        liveDeviceConfig: '3x1',
        liveDeviceBw: 80,
        liveDeviceMac: '98:EE:94:99:D4:1A',
        liveDeviceMimo: 'SU',
        liveCapturing: false,

        // _chartInstance is stored outside Alpine scope to avoid Proxy recursion

        init() {
            this._connect();
            var self = this;
            var timer = setInterval(function() {
                var canvas = document.getElementById('historyChart');
                if (!canvas) return;
                clearInterval(timer);
                self._initChart();
            }, 50);
        },

        _initChart() {
            var canvas = document.getElementById('historyChart');
            if (!canvas || _chartInstance) return;
            var container = canvas.parentElement;
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            _chartInstance = new Chart(canvas, {
                type: 'scatter',
                data: {
                    datasets: [
                        {
                            label: 'Empty',
                            data: [],
                            backgroundColor: 'rgba(6, 182, 212, 0.6)',
                            pointRadius: 4,
                            pointHoverRadius: 7,
                        },
                        {
                            label: 'Occupied',
                            data: [],
                            backgroundColor: 'rgba(249, 115, 22, 0.6)',
                            pointRadius: 4,
                            pointHoverRadius: 7,
                        },
                    ],
                },
                options: {
                    responsive: false,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: { mode: 'nearest', intersect: true },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            enabled: true,
                            backgroundColor: 'rgba(17, 24, 39, 0.95)',
                            titleColor: '#d1d5db',
                            bodyColor: '#9ca3af',
                            borderColor: '#374151',
                            borderWidth: 1,
                            padding: 10,
                            callbacks: {
                                title: function(items) {
                                    return items.length ? items[0].raw.x.toFixed(2) + 's' : '';
                                },
                                label: function(item) {
                                    return item.dataset.label + ': ' + (item.parsed.y * 100).toFixed(1) + '%';
                                },
                                afterLabel: function(item) {
                                    var r = item.raw;
                                    var lines = [];
                                    if (r.gt) lines.push('True: ' + r.gt);
                                    if (r.file) lines.push(r.file);
                                    return lines.join('\n');
                                },
                            },
                        },
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'Time (s)', color: '#9ca3af' },
                            ticks: { color: '#6b7280', maxTicksLimit: 12 },
                            grid: { color: '#374151' },
                            min: 0,
                        },
                        y: {
                            title: { display: true, text: 'P(Occupied)', color: '#9ca3af' },
                            ticks: { color: '#6b7280' },
                            grid: { color: '#374151' },
                            min: 0,
                            max: 1,
                        },
                    },
                },
            });
            var self = this;
            window.addEventListener('resize', function() {
                if (!_chartInstance || !container) return;
                canvas.width = container.clientWidth;
                canvas.height = container.clientHeight;
                _chartInstance.resize();
            });
            if (this.displayData.length > 0) {
                this._drawChart();
            }
        },

        _connect() {
            var self = this;
            var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            var ws = new WebSocket(proto + '//' + location.host + '/ws/predictions');

            ws.onopen = function() { self.connected = true; };
            ws.onclose = function() {
                self.connected = false;
                setTimeout(function() { self._connect(); }, 2000);
            };
            ws.onmessage = function(event) {
                self._handleMessage(JSON.parse(event.data));
            };
        },

        _handleMessage(msg) {
            switch (msg.type) {
                case 'prediction': this._onPrediction(msg.data); break;
                case 'state': this._onState(msg.data); break;
                case 'replay_data': this._onReplayData(msg.data); break;
                case 'log': this._onLog(msg.data); break;
            }
        },

        _onPrediction(data) {
            if (this.mode !== 'live') return;
            this.allPredictions.push(data);
            this.totalWindows = this.allPredictions.length;
            this.current = data;
            if (_chartInstance) {
                var pt = { x: this.totalWindows, y: data.p_occupied };
                var ds = data.ground_truth === 'empty' ? 0 : 1;
                _chartInstance.data.datasets[ds].data.push(pt);
                _chartInstance.update('none');
            }
        },

        _onState(data) {
            if (data.mode) this.mode = data.mode;
            if (data.replay_total_windows) this.totalWindows = data.replay_total_windows;
            if (data.live_capturing !== undefined) this.liveCapturing = data.live_capturing;
        },

        _onReplayData(data) {
            this.allPredictions = data.predictions;
            this.totalWindows = data.predictions.length;

            var fileSet = new Set();
            for (var i = 0; i < data.predictions.length; i++) {
                if (data.predictions[i].file) fileSet.add(data.predictions[i].file);
            }
            this.availableFiles = Array.from(fileSet).sort();
            this.selectedFile = 'all';
            this.filteredPredictions = data.predictions;
            this._rebuildDisplay();
            console.log('replay_data received:', data.predictions.length, 'predictions, chart:', !!_chartInstance);
        },

        _onLog(entry) {
            entry.id = this.logId++;
            this.logEntries.push(entry);
            if (this.logEntries.length > 200) this.logEntries.shift();
            this.$nextTick(function() {
                var el = this.$refs.logContainer;
                if (el) el.scrollTop = el.scrollHeight;
            }.bind(this));
        },

        switchMode(mode) {
            if (this.mode === mode) return;
            this._stopPlayback();
            this.mode = mode;
            this.current = {
                prediction: null, confidence: null, p_empty: null, p_occupied: null,
                dist_empty: null, dist_occupied: null, ground_truth: null,
                replay_file: null, device: null
            };
            if (mode === 'replay') {
                if (this.allPredictions.length > 0) {
                    this.filteredPredictions = this.allPredictions;
                    this.selectedFile = 'all';
                    this._rebuildDisplay();
                } else {
                    this.displayData = [];
                    this.totalTime = 0;
                    this._resetChart();
                }
            } else {
                this.displayData = [];
                this.totalTime = 0;
                this.currentTime = 0;
                this.currentIdx = 0;
                this._resetChart();
            }
            ws?.send(JSON.stringify({ type: 'mode', data: { mode: mode } }));
        },

        selectFile(file) {
            this._stopPlayback();
            this.selectedFile = file;
            if (file === 'all') {
                this.filteredPredictions = this.allPredictions;
            } else {
                var f = file;
                this.filteredPredictions = this.allPredictions.filter(function(p) { return p.file === f; });
            }
            this._rebuildDisplay();
        },

        _rebuildDisplay() {
            var preds = this.filteredPredictions;
            if (!preds.length) {
                this.displayData = [];
                this.totalTime = 0;
                this.currentTime = 0;
                this.currentIdx = 0;
                this.current = {
                    prediction: null, confidence: null, p_empty: null, p_occupied: null,
                    dist_empty: null, dist_occupied: null, ground_truth: null,
                    replay_file: null, device: null
                };
                this._resetChart();
                return;
            }
            var firstTs = preds[0].timestamp_s;
            this.displayData = preds.map(function(p) {
                return Object.assign({}, p, { dt: p.timestamp_s - firstTs });
            });
            this.totalTime = this.displayData[this.displayData.length - 1].dt;
            this.currentTime = 0;
            this.currentIdx = 0;
            this.current = Object.assign({}, this.displayData[0]);
            this._drawChart();
        },

        seekTo(t) {
            this._stopPlayback();
            this.currentTime = t;
            this._findAndSetPrediction(t);
        },

        _findAndSetPrediction(t) {
            var d = this.displayData;
            if (!d.length) return;
            var lo = 0, hi = d.length - 1;
            while (lo < hi) {
                var mid = (lo + hi) >> 1;
                if (d[mid].dt < t) lo = mid + 1;
                else hi = mid;
            }
            this._setPrediction(lo);
        },

        _setPrediction(idx) {
            var d = this.displayData;
            if (idx < 0 || idx >= d.length) return;
            this.currentIdx = idx;
            this.current = Object.assign({}, d[idx]);
            this.currentTime = d[idx].dt;
        },

        togglePlay() {
            if (this.playing) this._stopPlayback();
            else this._startPlayback();
        },

        _startPlayback() {
            this.playing = true;
            this._lastFrameTime = performance.now();
            this._tick();
        },

        _stopPlayback() {
            this.playing = false;
            if (this._animFrame) {
                cancelAnimationFrame(this._animFrame);
                this._animFrame = null;
            }
            this._lastFrameTime = null;
        },

        _tick() {
            if (!this.playing || !this.displayData.length) return;
            var now = performance.now();
            var dt = (now - this._lastFrameTime) / 1000;
            this._lastFrameTime = now;
            this.currentTime += dt * this.speed;
            if (this.currentTime >= this.totalTime) this.currentTime = 0;
            this._findAndSetPrediction(this.currentTime);
            this._syncVideo();
            this._animFrame = requestAnimationFrame(this._tick.bind(this));
        },

        setSpeed(s) { this.speed = s; },

        _syncVideo() {
            if (!this.showVideo || !this.videoSrc) return;
            var video = document.getElementById('syncVideo');
            if (!video) return;
            var target = this.currentTime + this.videoOffsetMs / 1000;
            if (Math.abs(video.currentTime - target) > 0.1) video.currentTime = target;
        },

        onVideoTimeUpdate() {},

        liveStart() {
            ws?.send(JSON.stringify({ type: 'live', data: {
                action: 'start', interface: this.liveInterface,
                device_name: this.liveDeviceName, standard: this.liveDeviceStandard,
                config: this.liveDeviceConfig, bw: this.liveDeviceBw,
                mac: this.liveDeviceMac, mimo: this.liveDeviceMimo,
            }}));
            this.liveCapturing = true;
        },

        liveStop() {
            ws?.send(JSON.stringify({ type: 'live', data: { action: 'stop' } }));
            this.liveCapturing = false;
        },

        liveReset() {
            this.liveStop();
            this.allPredictions = [];
            this.displayData = [];
            this.totalWindows = 0;
            this.totalTime = 0;
            this.currentTime = 0;
            this.currentIdx = 0;
            this.current = {
                prediction: null, confidence: null, p_empty: null, p_occupied: null,
                dist_empty: null, dist_occupied: null, ground_truth: null,
                replay_file: null, device: null
            };
            this._resetChart();
        },

        _drawChart() {
            if (!_chartInstance || !this.displayData.length) return;
            var emptyData = [], occData = [];
            for (var i = 0; i < this.displayData.length; i++) {
                var p = this.displayData[i];
                var pt = { x: p.dt, y: p.p_occupied, gt: p.ground_truth, file: p.file };
                if (p.ground_truth === 'empty') emptyData.push(pt);
                else occData.push(pt);
            }
            _chartInstance.data.datasets[0].data = emptyData;
            _chartInstance.data.datasets[1].data = occData;
            _chartInstance.options.scales.x.min = 0;
            _chartInstance.options.scales.x.max = this.totalTime || 1;
            _chartInstance.update('none');
        },

        _resetChart() {
            if (!_chartInstance) return;
            _chartInstance.data.datasets[0].data = [];
            _chartInstance.data.datasets[1].data = [];
            _chartInstance.options.scales.x.min = 0;
            _chartInstance.options.scales.x.max = 1;
            _chartInstance.update('none');
        },

        formatTime(s) {
            if (!s || !isFinite(s)) return '0:00';
            var mins = Math.floor(s / 60);
            var secs = Math.floor(s % 60);
            return mins + ':' + (secs < 10 ? '0' : '') + secs;
        },

        get windowDisplay() {
            if (!this.displayData.length) return '---';
            return 'Window ' + (this.currentIdx + 1) + ' of ' + this.displayData.length;
        },

        get groundTruthLabel() {
            var gt = this.current.ground_truth;
            if (!gt) return '';
            if (gt === 'empty') return 'empty';
            if (gt === 'stationary' || gt === 'moving') return 'occupied';
            return gt;
        },
    };
}
