/**
 * AI婴儿床监护系统 - 网页端Demo
 * YOLO危险动作检测可视化
 * 
 * 使用预计算的YOLO检测数据，检测框精确跟随婴儿身体移动
 */

// Web-accessible copy of data/442655__josephvm__baby-girl-crying.wav.
const CRYING_TEST_AUDIO = 'assets/audio/baby_crying.wav';
const CRYING_TRIGGER_THRESHOLD_DB = 55;

class BabyMonitorDemo {
    constructor() {
        this.videoPlayer = document.getElementById('videoPlayer');
        this.canvas = document.getElementById('overlayCanvas');
        this.ctx = this.canvas.getContext('2d');

        // 状态元素
        this.statusOverlay = document.getElementById('statusOverlay');
        this.statusBadge = document.getElementById('statusBadge');
        this.dangerOverlay = document.getElementById('dangerOverlay');

        // 信息显示元素
        this.currentStageEl = document.getElementById('currentStage');
        this.rawStageEl = document.getElementById('rawStage');
        this.confidenceEl = document.getElementById('confidence');
        this.targetPosEl = document.getElementById('targetPos');
        this.poseModeEl = document.getElementById('poseMode');
        this.visibleKeypointsEl = document.getElementById('visibleKeypoints');
        this.poseRiskScoreEl = document.getElementById('poseRiskScore');
        this.evidenceWristsEl = document.getElementById('evidenceWrists');
        this.evidenceShouldersEl = document.getElementById('evidenceShoulders');
        this.evidenceLeanEl = document.getElementById('evidenceLean');
        this.evidenceOutsideEl = document.getElementById('evidenceOutside');

        // 统计元素
        this.warningCountEl = document.getElementById('warningCount');
        this.dangerCountEl = document.getElementById('dangerCount');
        this.eventCountEl = document.getElementById('eventCount');

        // 控制元素
        this.playBtn = document.getElementById('playBtn');
        this.pauseBtn = document.getElementById('pauseBtn');
        this.progressBar = document.getElementById('progressBar');
        this.currentTimeEl = document.getElementById('currentTime');
        this.totalTimeEl = document.getElementById('totalTime');
        this.videoSelect = document.getElementById('videoSelect');
        this.loadVideoBtn = document.getElementById('loadVideoBtn');

        // 事件列表
        this.eventsList = document.getElementById('eventsList');

        // Voice Companion Agent
        this.simulateCryingBtn = document.getElementById('simulateCryingBtn');
        this.playComfortBtn = document.getElementById('playComfortBtn');
        this.stopComfortBtn = document.getElementById('stopComfortBtn');
        this.recordComfortBtn = document.getElementById('recordComfortBtn');
        this.whiteNoiseToggle = document.getElementById('whiteNoiseToggle');
        this.voiceOptionButtons = [...document.querySelectorAll('.voice-option')];
        this.voiceData = null;
        this.selectedVoice = null;
        this.whiteNoiseContext = null;
        this.whiteNoiseSource = null;

        // 顶部 Demo 导航
        this.demoTabs = [...document.querySelectorAll('.demo-tab')];
        this.demoTabPanels = [...document.querySelectorAll('.demo-tab-panel')];

        // 数据存储
        this.detectionData = null;  // 预计算的YOLO检测数据
        this.framesByIndex = null;  // 按帧索引索引的Map，加速查找
        this.videoInfo = null;      // 视频信息（分辨率、fps等）
        this.geometry = null;       // 几何配置（安全区、警告区、危险边界）
        this.currentVideo = 'dangerous_test6';
        this.events = [];
        this.triggeredEventKeys = new Set();
        this._warningIndex = 0;
        this._audioManifest = null;

        this.init();
    }

    init() {
        window._demo = this;
        this.setupEventListeners();
        this.setupDemoTabs();
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        this.loadGrowthMemory();
        this.loadVoiceCompanion();
        this.videoSelect.value = this.currentVideo;
        this.loadVideo(this.currentVideo);
        // 预加载语音列表（部分浏览器异步加载）
        if ('speechSynthesis' in window) {
            window.speechSynthesis.getVoices();
            window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
        }
    }

    setupDemoTabs() {
        for (const tab of this.demoTabs) {
            tab.addEventListener('click', () => this.activateDemoTab(tab.dataset.tabTarget));
        }
    }

    activateDemoTab(targetId) {
        for (const tab of this.demoTabs) {
            const isActive = tab.dataset.tabTarget === targetId;
            tab.classList.toggle('active', isActive);
            tab.setAttribute('aria-selected', String(isActive));
        }

        for (const panel of this.demoTabPanels) {
            panel.classList.toggle('active', panel.id === targetId);
        }

        if (targetId === 'voiceCompanionTab') {
            this.videoPlayer.pause();
        } else {
            this.stopComfortPlayback();
            requestAnimationFrame(() => this.resizeCanvas());
        }

        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    setupEventListeners() {
        this.playBtn.addEventListener('click', () => this.videoPlayer.play());
        this.pauseBtn.addEventListener('click', () => this.videoPlayer.pause());

        this.progressBar.addEventListener('input', (e) => {
            const time = (e.target.value / 100) * this.videoPlayer.duration;
            this.videoPlayer.currentTime = time;
        });

        this.videoPlayer.addEventListener('timeupdate', () => this.onTimeUpdate());
        this.videoPlayer.addEventListener('loadedmetadata', () => this.onLoadedMetadata());
        this.videoPlayer.addEventListener('play', () => this.animate());
        this.videoPlayer.addEventListener('seeked', () => this.render());

        this.loadVideoBtn.addEventListener('click', () => {
            this.currentVideo = this.videoSelect.value;
            this.loadVideo(this.currentVideo);
        });

        this.simulateCryingBtn.addEventListener('click', () => this.simulateCryingEvent());
        this.playComfortBtn.addEventListener('click', () => this.playComfortSpeech());
        this.stopComfortBtn.addEventListener('click', () => this.stopComfortPlayback());
        this.recordComfortBtn.addEventListener('click', () => this.recordComfortResult());
        this.whiteNoiseToggle.addEventListener('change', () => {
            if (this.whiteNoiseToggle.checked) this.startWhiteNoise();
            else this.stopWhiteNoise();
        });
        for (const button of this.voiceOptionButtons) {
            button.addEventListener('click', () => this.selectVoiceOption(button.dataset.voice, true));
        }
    }

    resizeCanvas() {
        // Canvas需要精确匹配视频在 object-fit:contain 下的实际显示区域
        const container = this.videoPlayer.parentElement;
        const containerW = container.clientWidth;
        const containerH = container.clientHeight;

        if (this.videoInfo) {
            // 有视频信息时，计算contain模式下的实际显示区域
            const videoAspect = this.videoInfo.width / this.videoInfo.height;
            const containerAspect = containerW / containerH;

            if (videoAspect > containerAspect) {
                // 视频更宽，左右撑满，上下有黑边
                this.canvas.width = containerW;
                this.canvas.height = containerW / videoAspect;
            } else {
                // 视频更高，上下撑满，左右有黑边
                this.canvas.height = containerH;
                this.canvas.width = containerH * videoAspect;
            }

            // 居中Canvas
            this.canvas.style.position = 'absolute';
            this.canvas.style.left = ((containerW - this.canvas.width) / 2) + 'px';
            this.canvas.style.top = ((containerH - this.canvas.height) / 2) + 'px';
        } else {
            // 没有视频信息时，使用容器尺寸
            this.canvas.width = containerW;
            this.canvas.height = containerH;
        }
    }

    /**
     * 加载视频和对应的预计算检测数据
     */
    async loadVideo(videoName) {
        // 重置状态
        this.events = [];
        this.triggeredEventKeys.clear();
        this.detectionData = null;
        this.framesByIndex = null;
        this.geometry = null;
        this.updateEventsList();
        this.updateStats(0, 0, 0);
        this.dangerOverlay.classList.add('hidden');
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.updatePoseEvidence(null);
        this.updateVideoDisclosure(videoName);

        // 加载视频
        this.videoPlayer.src = `data/${videoName}.mp4`;
        this.videoPlayer.load();

        // 加载预计算的YOLO检测数据
        try {
            const response = await fetch(`data/${videoName}_detection.json`);
            if (response.ok) {
                this.detectionData = await response.json();
                this.videoInfo = this.detectionData.video_info;
                this.geometry = this.detectionData.geometry;

                // 构建帧索引Map，加速查找（O(1)查找）
                this.framesByIndex = new Map();
                for (const frame of this.detectionData.frames) {
                    this.framesByIndex.set(frame.frame_index, frame);
                }

                console.log(`Detection data loaded: ${this.detectionData.frames.length} frames, ${this.detectionData.events.length} events`);
                console.log(`Video: ${this.videoInfo.width}x${this.videoInfo.height}, ${this.videoInfo.fps}fps, ${this.videoInfo.duration}s`);
                console.log(`Geometry: safe_zone=${this.geometry.safe_zone}, warning_zone=${this.geometry.warning_zone}, danger_x=${this.geometry.danger_boundary_x}`);

                // 重新计算Canvas尺寸（现在有视频分辨率信息了）
                this.resizeCanvas();
            } else {
                console.error('Detection data not found:', response.status);
                this.showNoDataMessage();
            }
        } catch (error) {
            console.error('Failed to load detection data:', error);
            this.showNoDataMessage();
        }

        // 视频加载错误处理
        this.videoPlayer.onerror = () => {
            console.error('Failed to load video:', this.videoPlayer.src);
            alert('视频加载失败，请确认视频文件存在于 web_demo/data/ 目录下');
        };
    }

    showNoDataMessage() {
        this.eventsList.innerHTML = '<div class="event-item empty">未找到预计算的检测数据，请先运行 generate_web_demo_data.py</div>';
    }

    updateVideoDisclosure(videoName) {
        const disclosure = document.getElementById('videoDisclosure');
        if (!disclosure) return;

        if (videoName === 'dangerous_test6') {
            disclosure.innerHTML = `
                <span class="disclosure-badge synthetic">AI 合成场景素材</span>
                <span>检测框、姿态关键点与状态数据由 YOLO11n-pose 离线推理生成，不代表真实环境准确率。</span>
            `;
        } else {
            disclosure.innerHTML = `
                <span class="disclosure-badge">本地测试素材</span>
                <span>检测框与姿态关键点由 YOLO11n-pose 预计算，用于稳定复现风险判断链路。</span>
            `;
        }
    }

    onLoadedMetadata() {
        this.totalTimeEl.textContent = this.formatTime(this.videoPlayer.duration);
        this.progressBar.max = 100;
    }

    onTimeUpdate() {
        const progress = (this.videoPlayer.currentTime / this.videoPlayer.duration) * 100;
        this.progressBar.value = progress;
        this.currentTimeEl.textContent = this.formatTime(this.videoPlayer.currentTime);
    }

    animate() {
        if (this.videoPlayer.paused || this.videoPlayer.ended) return;
        this.render();
        requestAnimationFrame(() => this.animate());
    }

    /**
     * 核心渲染：每帧调用，绘制检测区域和检测框
     */
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (!this.detectionData || !this.videoInfo) return;

        // 根据当前视频时间计算帧索引
        const currentTime = this.videoPlayer.currentTime;
        const fps = this.videoInfo.fps;
        const frameIndex = Math.round(currentTime * fps);

        // 从预计算数据中精确查找当前帧
        const frameData = this.framesByIndex.get(frameIndex);
        if (!frameData) return;

        // 计算视频原始分辨率到Canvas显示尺寸的缩放比例
        const scaleX = this.canvas.width / this.videoInfo.width;
        const scaleY = this.canvas.height / this.videoInfo.height;

        // 1. 绘制检测区域（安全区、警告区、危险边界）
        this.drawZones(scaleX, scaleY);

        // 2. 绘制YOLO检测框（坐标从原始分辨率映射到Canvas尺寸）
        if (frameData.target) {
            this.drawDetection(frameData.target, frameData.stage, scaleX, scaleY);
        }

        // 3. 更新右侧信息面板
        this.updateStatus(frameData);

        // 4. 检查危险事件触发
        this.checkEvents(frameData);
    }

    /**
     * 绘制安全区、警告区、危险边界
     * 支持多边形模式（crib_contour/safe_contour）和矩形模式（兼容旧数据）
     */
    drawZones(scaleX, scaleY) {
        const geo = this.geometry;
        if (!geo) return;

        // ── 多边形模式 ──
        if (geo.crib_contour && geo.safe_contour) {
            // 婴儿床边界（= warning zone，红色多边形）
            this._drawPolygon(geo.crib_contour, scaleX, scaleY,
                'rgba(248, 113, 113, 0.7)', 'rgba(248, 113, 113, 0.06)', 3);
            this.ctx.fillStyle = 'rgba(248, 113, 113, 0.9)';
            this.ctx.font = `bold ${Math.max(12, 14 * scaleX)}px sans-serif`;
            const cp0 = geo.crib_contour[0];
            this.ctx.fillText('婴儿床边界 CRIB BOUNDARY', cp0[0] * scaleX + 8, cp0[1] * scaleY + 18);

            // 安全区（绿色缩小多边形）
            this._drawPolygon(geo.safe_contour, scaleX, scaleY,
                'rgba(74, 222, 128, 0.8)', 'rgba(74, 222, 128, 0.08)', 2);
            this.ctx.fillStyle = 'rgba(74, 222, 128, 0.9)';
            this.ctx.font = `${Math.max(12, 14 * scaleX)}px sans-serif`;
            const sp0 = geo.safe_contour[0];
            this.ctx.fillText('安全区域 SAFE ZONE', sp0[0] * scaleX + 8, sp0[1] * scaleY + 18);

            return;
        }

        // ── 矩形模式（兼容旧数据）──
        if (!geo.safe_zone || !geo.warning_zone) return;

        const [sx, sy, sw, sh] = geo.safe_zone;
        const [wx, wy, ww, wh] = geo.warning_zone;
        const dangerX = geo.danger_boundary_x;

        const safeX = sx * scaleX, safeY = sy * scaleY, safeW = sw * scaleX, safeH = sh * scaleY;
        const warnX = wx * scaleX, warnY = wy * scaleY, warnW = ww * scaleX, warnH = wh * scaleY;

        // 安全区
        this.ctx.strokeStyle = 'rgba(74, 222, 128, 0.8)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([8, 4]);
        this.ctx.strokeRect(safeX, safeY, safeW, safeH);
        this.ctx.setLineDash([]);
        this.ctx.fillStyle = 'rgba(74, 222, 128, 0.08)';
        this.ctx.fillRect(safeX, safeY, safeW, safeH);
        this.ctx.fillStyle = 'rgba(74, 222, 128, 0.9)';
        this.ctx.font = `${Math.max(12, 14 * scaleX)}px sans-serif`;
        this.ctx.fillText('安全区域 SAFE ZONE', safeX + 8, safeY + 20 * scaleY);

        // 警告区
        this.ctx.strokeStyle = 'rgba(251, 191, 36, 0.8)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([8, 4]);
        this.ctx.strokeRect(warnX, warnY, warnW, warnH);
        this.ctx.setLineDash([]);
        this.ctx.fillStyle = 'rgba(251, 191, 36, 0.08)';
        this.ctx.fillRect(warnX, warnY, warnW, warnH);
        this.ctx.fillStyle = 'rgba(251, 191, 36, 0.9)';
        this.ctx.fillText('关注区域 WARNING ZONE', warnX + 8, warnY + 20 * scaleY);

        // 危险边界线
        if (dangerX) {
            const dangerXCanvas = dangerX * scaleX;
            this.ctx.strokeStyle = 'rgba(248, 113, 113, 0.9)';
            this.ctx.lineWidth = 3;
            this.ctx.setLineDash([10, 5]);
            this.ctx.beginPath();
            this.ctx.moveTo(dangerXCanvas, 0);
            this.ctx.lineTo(dangerXCanvas, this.canvas.height);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            this.ctx.fillStyle = 'rgba(248, 113, 113, 0.9)';
            this.ctx.font = `bold ${Math.max(12, 14 * scaleX)}px sans-serif`;
            this.ctx.fillText('危险边界 DANGER BOUNDARY', dangerXCanvas - 150 * scaleX, 25 * scaleY);
        }
    }

    /**
     * 绘制多边形轮廓（用于婴儿床不规则边界）
     * points 格式: [[x1,y1], [x2,y2], ...]
     */
    _drawPolygon(points, scaleX, scaleY, strokeColor, fillColor, lineWidth) {
        if (!points || points.length < 3) return;
        const ctx = this.ctx;
        ctx.beginPath();
        ctx.moveTo(points[0][0] * scaleX, points[0][1] * scaleY);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0] * scaleX, points[i][1] * scaleY);
        }
        ctx.closePath();
        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash([8, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    /**
     * 绘制YOLO检测框
     * 坐标从视频原始分辨率映射到Canvas显示尺寸
     */
    drawDetection(target, stage, scaleX, scaleY) {
        const { x, y, w, h } = target.bbox;
        const { x: cx, y: cy } = target.center;

        // 映射到Canvas坐标
        const bx = x * scaleX, by = y * scaleY;
        const bw = w * scaleX, bh = h * scaleY;
        const bcx = cx * scaleX, bcy = cy * scaleY;

        // 根据状态选择颜色
        let color, bgColor;
        if (stage === 'DANGEROUS_ACTION') {
            color = '#f87171';
            bgColor = 'rgba(248, 113, 113, 0.15)';
        } else if (stage === 'WARNING') {
            color = '#fbbf24';
            bgColor = 'rgba(251, 191, 36, 0.1)';
        } else {
            color = '#4ade80';
            bgColor = 'rgba(74, 222, 128, 0.08)';
        }

        // 检测框背景填充
        this.ctx.fillStyle = bgColor;
        this.ctx.fillRect(bx, by, bw, bh);

        this.drawPoseSkeleton(target, scaleX, scaleY);

        // 检测框边框
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(bx, by, bw, bh);

        // 四角标记（更醒目的检测框样式）
        const cornerLen = Math.min(20, bw * 0.15, bh * 0.15);
        this.ctx.lineWidth = 4;
        // 左上
        this.ctx.beginPath();
        this.ctx.moveTo(bx, by + cornerLen); this.ctx.lineTo(bx, by); this.ctx.lineTo(bx + cornerLen, by);
        this.ctx.stroke();
        // 右上
        this.ctx.beginPath();
        this.ctx.moveTo(bx + bw - cornerLen, by); this.ctx.lineTo(bx + bw, by); this.ctx.lineTo(bx + bw, by + cornerLen);
        this.ctx.stroke();
        // 左下
        this.ctx.beginPath();
        this.ctx.moveTo(bx, by + bh - cornerLen); this.ctx.lineTo(bx, by + bh); this.ctx.lineTo(bx + cornerLen, by + bh);
        this.ctx.stroke();
        // 右下
        this.ctx.beginPath();
        this.ctx.moveTo(bx + bw - cornerLen, by + bh); this.ctx.lineTo(bx + bw, by + bh); this.ctx.lineTo(bx + bw, by + bh - cornerLen);
        this.ctx.stroke();

        // 中心点（婴儿位置跟踪点）
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.arc(bcx, bcy, 5, 0, Math.PI * 2);
        this.ctx.fill();

        // 中心点外圈（脉冲效果）
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(bcx, bcy, 12, 0, Math.PI * 2);
        this.ctx.stroke();

        // 标签背景
        const targetLabel = target.class_name === 'person'
            ? '人体目标 PERSON'
            : String(target.class_name || 'TARGET').toUpperCase();
        const label = `${targetLabel}  ${(target.confidence * 100).toFixed(1)}%`;
        this.ctx.font = `bold ${Math.max(12, 14 * scaleX)}px monospace`;
        const textWidth = this.ctx.measureText(label).width;
        const labelX = bx;
        const labelY = Math.max(22 * scaleY, by - 8);

        this.ctx.fillStyle = color;
        this.ctx.fillRect(labelX, labelY - 16 * scaleY, textWidth + 12, 20 * scaleY);

        // 标签文字
        this.ctx.fillStyle = stage === 'WARNING' ? '#1a1a2e' : '#ffffff';
        this.ctx.fillText(label, labelX + 6, labelY - 2 * scaleY);

        // 如果是危险状态，绘制bbox_right触发线
        if (stage === 'DANGEROUS_ACTION' || stage === 'WARNING') {
            const bboxRight = (x + w) * scaleX;
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([4, 4]);
            this.ctx.beginPath();
            this.ctx.moveTo(bboxRight, by);
            this.ctx.lineTo(bboxRight, by + bh);
            this.ctx.stroke();
            this.ctx.setLineDash([]);

            // 目标框右边界标签
            this.ctx.fillStyle = color;
            this.ctx.font = `${Math.max(10, 11 * scaleX)}px sans-serif`;
            this.ctx.fillText(`目标右边界 BBOX RIGHT=${Math.round(x + w)}`, bboxRight + 5, by + bh / 2);
        }
    }

    drawPoseSkeleton(target, scaleX, scaleY) {
        const points = target.keypoints;
        if (!Array.isArray(points) || points.length < 17) return;

        const minConfidence = 0.3;
        const connections = [
            [5, 6],
            [5, 7], [7, 9],
            [6, 8], [8, 10],
            [5, 11], [6, 12],
            [11, 12],
            [11, 13], [13, 15],
            [12, 14], [14, 16],
        ];
        const visible = index => points[index] && points[index].confidence >= minConfidence;

        this.ctx.save();
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = 'rgba(226, 232, 240, 0.58)';
        for (const [from, to] of connections) {
            if (!visible(from) || !visible(to)) continue;
            this.ctx.beginPath();
            this.ctx.moveTo(points[from].x * scaleX, points[from].y * scaleY);
            this.ctx.lineTo(points[to].x * scaleX, points[to].y * scaleY);
            this.ctx.stroke();
        }

        points.forEach((point, index) => {
            if (!visible(index)) return;
            let fill = 'rgba(226, 232, 240, 0.75)';
            if ([9, 10].includes(index)) fill = '#facc15';
            else if ([5, 6].includes(index)) fill = '#fb923c';
            else if ([11, 12].includes(index)) fill = '#60a5fa';

            const radius = [5, 6, 9, 10, 11, 12].includes(index) ? 5 : 3;
            this.ctx.fillStyle = fill;
            this.ctx.beginPath();
            this.ctx.arc(point.x * scaleX, point.y * scaleY, radius, 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.restore();
    }

    getStageLabel(stage) {
        const labels = {
            SAFE: '安全 SAFE',
            WARNING: '关注 WARNING',
            DANGEROUS_ACTION: '危险 DANGEROUS',
        };
        return labels[stage] || stage || '--';
    }

    setStageClass(element, stage) {
        element.className = 'value';
        if (stage === 'SAFE') element.classList.add('stage-safe');
        else if (stage === 'WARNING') element.classList.add('stage-warning');
        else if (stage === 'DANGEROUS_ACTION') element.classList.add('stage-danger');
    }

    setEvidenceChip(element, label, active, detail = '') {
        element.textContent = `${label} ${active ? '是 YES' : '否 NO'}${detail}`;
        element.classList.toggle('active', Boolean(active));
    }

    updatePoseEvidence(analysis) {
        if (!analysis) {
            this.poseModeEl.textContent = '等待检测';
            this.poseModeEl.className = 'pose-mode';
            this.visibleKeypointsEl.textContent = '--/17';
            this.poseRiskScoreEl.textContent = '--/10';
            for (const [element, label] of [
                [this.evidenceWristsEl, '双手近护栏'],
                [this.evidenceShouldersEl, '肩部靠边'],
                [this.evidenceLeanEl, '上身前倾'],
                [this.evidenceOutsideEl, '关键点越界'],
            ]) {
                element.textContent = `${label} --`;
                element.classList.remove('active');
            }
            return;
        }

        const modeLabels = {
            pose_assisted: '姿态辅助 POSE',
            bbox_fallback: '框选降级 BBOX',
            no_target: '未检测目标',
        };
        this.poseModeEl.textContent = modeLabels[analysis.mode] || analysis.mode;
        this.poseModeEl.className = `pose-mode ${analysis.mode === 'pose_assisted' ? 'active' : 'fallback'}`;
        this.visibleKeypointsEl.textContent = `${analysis.visible_keypoints || 0}/17`;
        this.poseRiskScoreEl.textContent = `${analysis.score || 0}/10`;

        const evidence = analysis.evidence || {};
        this.setEvidenceChip(
            this.evidenceWristsEl,
            '双手近护栏',
            evidence.both_wrists_near_rail
        );
        this.setEvidenceChip(
            this.evidenceShouldersEl,
            '肩部靠边',
            evidence.shoulders_near_edge
        );
        this.setEvidenceChip(
            this.evidenceLeanEl,
            '上身前倾',
            evidence.upper_body_leaning_out
        );
        this.setEvidenceChip(
            this.evidenceOutsideEl,
            '关键点越界',
            Number(evidence.pose_points_outside || 0) > 0,
            ` (${Number(evidence.pose_points_outside || 0)})`
        );
    }

    /**
     * 更新右侧信息面板的状态显示
     */
    updateStatus(frameData) {
        const stage = frameData.stage;
        const rawStage = frameData.raw_stage;

        // 更新视频左上角状态标签
        this.statusOverlay.className = 'status-overlay';
        if (stage === 'SAFE') {
            this.statusOverlay.classList.add('safe');
            this.dangerOverlay.classList.add('hidden');
            this.videoPlayer.parentElement.classList.remove('danger-state');
        } else if (stage === 'WARNING') {
            this.statusOverlay.classList.add('warning');
            this.dangerOverlay.classList.add('hidden');
            this.videoPlayer.parentElement.classList.remove('danger-state');
        } else if (stage === 'DANGEROUS_ACTION') {
            this.statusOverlay.classList.add('danger');
            this.dangerOverlay.classList.remove('hidden');
            this.videoPlayer.parentElement.classList.add('danger-state');
        }
        this.statusBadge.textContent = this.getStageLabel(stage);

        // 更新右侧面板
        this.currentStageEl.textContent = this.getStageLabel(stage);
        this.setStageClass(this.currentStageEl, stage);
        this.rawStageEl.textContent = this.getStageLabel(rawStage);
        this.setStageClass(this.rawStageEl, rawStage);

        if (frameData.target) {
            this.confidenceEl.textContent = frameData.target.confidence.toFixed(4);
            this.targetPosEl.textContent = `(${Math.round(frameData.target.center.x)}, ${Math.round(frameData.target.center.y)})`;
        } else {
            this.confidenceEl.textContent = '--';
            this.targetPosEl.textContent = '--';
        }
        this.updatePoseEvidence(frameData.pose_analysis || null);

        this.updateStats(
            frameData.warning_count || 0,
            frameData.danger_count || 0,
            this.events.length
        );
    }

    /**
     * 检查危险事件触发（只在状态首次变为DANGEROUS_ACTION时记录）
     */
    checkEvents(frameData) {
        if (frameData.stage === 'DANGEROUS_ACTION' && frameData.target) {
            const matchedEvent = [...this.detectionData.events]
                .reverse()
                .find(e => e.frame_index <= frameData.frame_index);

            if (matchedEvent && !this.triggeredEventKeys.has('evt_' + matchedEvent.frame_index)) {
                this.triggeredEventKeys.add('evt_' + matchedEvent.frame_index);
                this.events.push({
                    timestamp: matchedEvent.timestamp,
                    stage: 'DANGEROUS_ACTION',
                    confidence: matchedEvent.confidence,
                    position: matchedEvent.target_center,
                    bbox: matchedEvent.target_bbox,
                });
                this.updateEventsList();
                this.eventCountEl.textContent = this.events.length;
                this.speakWarning();
            }
        }
    }

    /**
     * 语音播报：优先使用预生成的克隆音色音频，降级到 Web Speech API
     */
    speakWarning() {
        const warnings = [
            'assets/audio/danger_warning_1.wav',
            'assets/audio/danger_warning_2.wav',
            'assets/audio/danger_warning_3.wav',
        ];
        const fallbackText = '宝贝小心，请往中间来';
        const audioPath = warnings[this._warningIndex % warnings.length];
        this._warningIndex++;
        this.playClonedAudio(audioPath, fallbackText);
    }

    /**
     * 播放克隆音色音频，失败时降级到 Web Speech API
     */
    playClonedAudio(audioPath, fallbackText) {
        const audio = new Audio(audioPath);
        audio.volume = 0.9;
        audio.play().catch(() => {
            // 文件不存在或格式不支持 → 降级到 Web Speech API
            if (fallbackText && 'speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(fallbackText);
                utterance.lang = 'zh-CN';
                utterance.rate = 1.0;
                utterance.pitch = 1.1;
                utterance.volume = 0.9;
                const voices = window.speechSynthesis.getVoices();
                const zhVoice = voices.find(v => v.lang.startsWith('zh'));
                if (zhVoice) utterance.voice = zhVoice;
                window.speechSynthesis.speak(utterance);
            }
        });
        return audio;
    }

    updateStats(warningCount, dangerCount, eventCount) {
        this.warningCountEl.textContent = warningCount;
        this.dangerCountEl.textContent = dangerCount;
        this.eventCountEl.textContent = eventCount;
    }

    updateEventsList() {
        if (this.events.length === 0) {
            this.eventsList.innerHTML = '<div class="event-item empty">暂无事件</div>';
            return;
        }

        this.eventsList.innerHTML = this.events.map(event => `
            <div class="event-item">
                <div class="event-time">${this.formatTime(event.timestamp)}</div>
                <div class="event-desc">
                    危险事件 RISK EVENT | 置信度: ${(event.confidence * 100).toFixed(1)}% |
                    位置: (${Math.round(event.position.x)}, ${Math.round(event.position.y)})
                </div>
            </div>
        `).reverse().join('');
    }

    formatTime(seconds) {
        if (!seconds || isNaN(seconds)) return '00:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    // ── Growth Memory Agent ──────────────────────────────────

    async loadGrowthMemory() {
        const container = document.getElementById('memoryCardsContainer');
        // 显示思考状态动画
        container.innerHTML = `
            <div class="gm-thinking">
                <div class="gm-thinking-spinner">
                    <span></span><span></span><span></span>
                </div>
                <div class="gm-thinking-text">AI 正在分析宝宝成长数据...</div>
            </div>`;
        try {
            const response = await fetch('data/growth_memory.json');
            if (!response.ok) {
                container.innerHTML = '<div class="gm-empty">No memory data yet. Run <code>python generate_growth_memory_web_data.py</code> first.</div>';
                return;
            }
            const data = await response.json();
            this.growthMemoryData = data;
            const llmBadge = document.getElementById('llmBadge');
            if (llmBadge) {
                llmBadge.textContent = data.llm_used ? 'LLM 生成摘要' : '规则模板摘要';
            }

            this.renderSafetyScore(data.summary_stats);
            this.renderMemoryCards(data.memory_cards);
            this.renderTrendChart(data.trend_data);
            this.renderSuggestions(data.parent_suggestions);
        } catch (err) {
            console.error('Failed to load growth memory:', err);
            container.innerHTML = '<div class="gm-empty">No memory data yet. Run <code>python generate_growth_memory_web_data.py</code> first.</div>';
        }
    }

    renderSafetyScore(stats) {
        if (!stats) return;

        const score = stats.safety_score || 0;
        const scoreText = document.getElementById('safetyScoreText');
        const scoreCircle = document.getElementById('scoreCircle');
        const detailsEl = document.getElementById('scoreDetails');

        scoreText.textContent = score;

        // SVG circle: circumference = 2 * PI * 42 ≈ 263.9
        const circumference = 2 * Math.PI * 42;
        const offset = circumference * (1 - score / 100);

        let color = '#4ade80';
        if (score < 60) color = '#fbbf24';
        if (score < 40) color = '#f87171';

        scoreCircle.style.transition = 'stroke-dashoffset 1.5s ease, stroke 0.5s ease';
        scoreCircle.setAttribute('stroke', color);
        scoreCircle.setAttribute('stroke-dashoffset', String(offset));

        detailsEl.innerHTML = `
            <span class="score-detail-chip">覆盖 ${stats.days_covered} 天</span>
            <span class="score-detail-chip">${stats.total_events} 条事件</span>
            <span class="score-detail-chip">高风险 ${stats.high_risk_count} 次</span>
        `;
    }

    renderMemoryCards(cards) {
        const container = document.getElementById('memoryCardsContainer');
        if (!cards || cards.length === 0) {
            container.innerHTML = '<div class="gm-empty">暂无成长记忆</div>';
            return;
        }

        const iconMap = {
            exploration: '🔍',
            risk: '⚠️',
            tech: '🤖',
            empty: '📝',
        };

        container.innerHTML = cards.map(card => `
            <div class="memory-card-item severity-${card.severity || 'info'}">
                <div class="card-header">
                    <span class="card-title">${iconMap[card.icon] || '📝'} ${card.title}</span>
                    <span class="card-date">${card.date || ''}</span>
                </div>
                <div class="card-body">${card.body}</div>
                ${card.audio_file ? `<button class="voice-play-btn" data-audio="${card.audio_file}" onclick="window._demo.toggleCardAudio(this)">🔊 播放语音</button>` : ''}
            </div>
        `).join('');
    }

    renderTrendChart(trendData) {
        const container = document.getElementById('trendBars');
        if (!trendData || !trendData.daily_frequency) {
            container.innerHTML = '<div class="gm-empty">暂无趋势数据</div>';
            return;
        }

        const daily = trendData.daily_frequency;
        const entries = Object.entries(daily);
        if (entries.length === 0) {
            container.innerHTML = '<div class="gm-empty">暂无趋势数据</div>';
            return;
        }

        const maxVal = Math.max(...entries.map(([, v]) => v), 1);

        container.innerHTML = entries.map(([day, count]) => {
            const height = Math.max(4, (count / maxVal) * 70);
            const shortDay = day.slice(5); // "05-18"
            return `
                <div class="trend-bar-wrapper">
                    <span class="trend-bar-value">${count}</span>
                    <div class="trend-bar" style="height:${height}px"></div>
                    <span class="trend-bar-label">${shortDay}</span>
                </div>
            `;
        }).join('');
    }

    renderSuggestions(suggestions) {
        const container = document.getElementById('suggestionsList');
        if (!suggestions || suggestions.length === 0) {
            container.innerHTML = '<div class="gm-empty">暂无建议</div>';
            return;
        }

        container.innerHTML = suggestions.map(s => `
            <div class="suggestion-item priority-${s.priority || 'low'}">
                <div class="suggestion-title">${s.title}</div>
                <div class="suggestion-body">${s.body}</div>
                ${s.audio_file ? `<button class="voice-play-btn" data-audio="${s.audio_file}" onclick="window._demo.toggleCardAudio(this)">🔊 播放语音</button>` : ''}
            </div>
        `).join('');
    }

    /**
     * 切换记忆卡片/建议的语音播放（点击播放/停止）
     */
    toggleCardAudio(btn) {
        const audioPath = btn.dataset.audio;
        if (!audioPath) return;

        // 停止当前正在播放的音频
        if (this._currentCardAudio) {
            this._currentCardAudio.pause();
            this._currentCardAudio.currentTime = 0;
            const prevBtn = document.querySelector('.voice-play-btn.playing');
            if (prevBtn) prevBtn.classList.remove('playing');
            // 点击同一个按钮 = 停止
            if (this._currentCardAudio._btn === btn) {
                this._currentCardAudio = null;
                btn.textContent = '🔊 播放语音';
                return;
            }
            this._currentCardAudio = null;
        }

        const audio = new Audio(audioPath);
        audio._btn = btn;
        audio.volume = 0.9;
        btn.classList.add('playing');
        btn.textContent = '⏸ 播放中...';
        audio.play().then(() => {
            audio.onended = () => {
                btn.classList.remove('playing');
                btn.textContent = '🔊 播放语音';
                this._currentCardAudio = null;
            };
        }).catch(() => {
            btn.classList.remove('playing');
            btn.textContent = '🔊 播放语音';
            this._currentCardAudio = null;
        });
        this._currentCardAudio = audio;
    }

    // ── Voice Companion Agent ───────────────────────────────

    async loadVoiceCompanion() {
        try {
            const response = await fetch('data/voice_companion.json');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.voiceData = await response.json();
            document.getElementById('voiceScene').textContent =
                this.voiceData.crying_event.scene === 'night_sleep' ? '夜间入睡' : '白天互动';
            document.getElementById('cryIntensity').textContent =
                this.voiceData.crying_event.cry_intensity === 'medium' ? '中等 MEDIUM' : this.voiceData.crying_event.cry_intensity;
            this.simulateCryingBtn.disabled = false;
        } catch (error) {
            console.error('Failed to load Voice Companion data:', error);
            document.getElementById('comfortMemoryState').textContent =
                '未加载语音陪伴数据，请运行 generate_voice_companion_web_data.py';
            this.simulateCryingBtn.disabled = true;
        }
    }

    simulateCryingEvent() {
        if (!this.voiceData) return;

        // 播放哭声测试音频
        this._playCryingAudio();

        document.getElementById('cryingStatus').textContent = '检测到哭闹 CRYING';
        document.getElementById('cryingStatus').className = 'voice-alert';
        document.querySelector('.cry-panel').classList.add('completed');

        const memory = this.voiceData.matched_memory;
        document.getElementById('comfortMemoryState').classList.add('hidden');
        document.getElementById('comfortMemoryContent').classList.remove('hidden');
        document.getElementById('comfortMemorySummary').textContent = memory.memory_summary;
        document.getElementById('comfortSuccessRate').textContent =
            `${Math.round(memory.historical_success_rate * 100)}%`;
        document.getElementById('comfortEvidenceCount').textContent =
            `${memory.evidence_count} 次`;
        document.getElementById('memoryStrategy').textContent =
            `${memory.preferred_voice_label} + ${memory.preferred_background === 'white_noise' ? '低音量白噪音' : '无背景声'}`;
        document.querySelector('.memory-hit-panel').classList.add('completed');

        this.selectVoiceOption(this.voiceData.selected_voice, false);
        document.getElementById('comfortScript').textContent = this.voiceData.comfort_script;
        this.whiteNoiseToggle.checked = this.voiceData.background_audio === 'white_noise';
        this.whiteNoiseToggle.disabled = false;
        this.playComfortBtn.disabled = false;
        this.stopComfortBtn.disabled = false;
        this.recordComfortBtn.disabled = false;
        document.getElementById('comfortResult').textContent = '本次安抚结果：等待播放与模拟确认';
        document.getElementById('comfortResult').className = 'comfort-result pending';
        document.getElementById('comfortLogStatus').textContent = '尚未写入安抚记录';
        document.getElementById('comfortLogStatus').className = 'comfort-log-status';
    }

    selectVoiceOption(voiceId, manualSelection) {
        if (!this.voiceData) return;
        const option = this.voiceData.voice_options.find(item => item.id === voiceId);
        if (!option) return;

        this.selectedVoice = voiceId;
        for (const button of this.voiceOptionButtons) {
            button.classList.toggle('selected', button.dataset.voice === voiceId);
        }
        document.querySelector('.voice-strategy-panel').classList.add('completed');

        if (manualSelection) {
            const scripts = {
                mother: '宝宝，妈妈在这里。我们慢慢放松，准备睡觉。',
                father: '宝宝，爸爸在这里陪着你。我们慢慢安静下来。',
                default: '宝宝别怕，我在这里陪着你。',
            };
            document.getElementById('comfortScript').textContent = scripts[voiceId];
            this.whiteNoiseToggle.checked =
                voiceId === 'mother' && this.voiceData.crying_event.scene === 'night_sleep';
            if (!this.whiteNoiseToggle.checked) this.stopWhiteNoise();
        }
    }

    findRoleVoice(role) {
        const voices = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
        const chinese = voices.filter(voice => voice.lang && voice.lang.toLowerCase().startsWith('zh'));
        const keywords = role === 'mother'
            ? ['xiaoxiao', 'huihui', 'yaoyao', 'female', 'woman', '女']
            : ['yunxi', 'kangkang', 'male', 'man', '男'];
        return chinese.find(voice =>
            keywords.some(keyword => voice.name.toLowerCase().includes(keyword))
        ) || chinese[0] || voices[0] || null;
    }

    playComfortSpeech() {
        if (!this.voiceData || !this.selectedVoice) return;
        const script = document.getElementById('comfortScript').textContent;

        if (this.whiteNoiseToggle.checked) this.startWhiteNoise();
        if (!('speechSynthesis' in window)) {
            document.getElementById('comfortResult').textContent =
                '当前浏览器不支持 Web Speech API，已保留策略展示作为兜底。';
            document.getElementById('comfortResult').className = 'comfort-result warning';
            return;
        }

        window.speechSynthesis.cancel();
        const profiles = {
            mother: { rate: 0.82, pitch: 1.12, volume: 0.82 },
            father: { rate: 0.86, pitch: 0.82, volume: 0.86 },
            default: { rate: 0.92, pitch: 1.0, volume: 0.85 },
        };
        const profile = profiles[this.selectedVoice] || profiles.default;
        const utterance = new SpeechSynthesisUtterance(script);
        utterance.lang = 'zh-CN';
        utterance.rate = profile.rate;
        utterance.pitch = profile.pitch;
        utterance.volume = profile.volume;
        const voice = this.findRoleVoice(this.selectedVoice);
        if (voice) utterance.voice = voice;
        document.querySelector('.playback-panel').classList.add('active');
        document.getElementById('comfortResult').textContent =
            '已调用本地浏览器 TTS 播放模拟安抚语';
        document.getElementById('comfortResult').className = 'comfort-result playing';
        utterance.onstart = () => {
            document.querySelector('.playback-panel').classList.add('active');
            document.getElementById('comfortResult').textContent =
                '正在播放模拟安抚语（本地浏览器 TTS）';
            document.getElementById('comfortResult').className = 'comfort-result playing';
        };
        utterance.onend = () => {
            document.querySelector('.playback-panel').classList.remove('active');
            document.getElementById('comfortResult').textContent =
                '安抚语播放完成，等待模拟结果确认';
            document.getElementById('comfortResult').className = 'comfort-result pending';
        };
        window.speechSynthesis.speak(utterance);
    }

    stopComfortPlayback() {
        if ('speechSynthesis' in window) window.speechSynthesis.cancel();
        this.stopWhiteNoise();
        if (this._cryingAudio) {
            this._cryingAudio.pause();
            this._cryingAudio.currentTime = 0;
            this._cryingAudio = null;
        }
        document.querySelector('.playback-panel').classList.remove('active');
        document.getElementById('comfortResult').textContent = '播放已停止';
        document.getElementById('comfortResult').className = 'comfort-result pending';
    }

    _playCryingAudio() {
        if (this._cryingAudio) {
            this._cryingAudio.pause();
            this._cryingAudio.currentTime = 0;
        }
        const audio = new Audio(CRYING_TEST_AUDIO);
        audio.volume = 0.7;
        this._cryingAudio = audio;
        audio.play().then(() => {
            // 更新 audio-slot 状态
            const slots = document.querySelectorAll('.audio-slot strong');
            if (slots[0]) slots[0].textContent = '正在播放哭声测试';
        }).catch(() => {
            const slots = document.querySelectorAll('.audio-slot strong');
            if (slots[0]) slots[0].textContent = '音频加载失败';
        });
        audio.onended = () => {
            const slots = document.querySelectorAll('.audio-slot strong');
            if (slots[0]) slots[0].textContent = '哭声播放完毕';
            this._cryingAudio = null;
        };
    }

    startWhiteNoise() {
        if (this.whiteNoiseSource) return;
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return;
        this.whiteNoiseContext = this.whiteNoiseContext || new AudioContextClass();
        const bufferSize = this.whiteNoiseContext.sampleRate * 2;
        const buffer = this.whiteNoiseContext.createBuffer(1, bufferSize, this.whiteNoiseContext.sampleRate);
        const output = buffer.getChannelData(0);
        for (let index = 0; index < bufferSize; index += 1) {
            output[index] = (Math.random() * 2 - 1) * 0.12;
        }
        const source = this.whiteNoiseContext.createBufferSource();
        const gain = this.whiteNoiseContext.createGain();
        gain.gain.value = 0.055;
        source.buffer = buffer;
        source.loop = true;
        source.connect(gain).connect(this.whiteNoiseContext.destination);
        source.start();
        this.whiteNoiseSource = source;
    }

    stopWhiteNoise() {
        if (!this.whiteNoiseSource) return;
        try {
            this.whiteNoiseSource.stop();
        } catch (error) {
            console.debug('White noise source already stopped', error);
        }
        this.whiteNoiseSource = null;
    }

    async recordComfortResult() {
        if (!this.voiceData || !this.selectedVoice) return;
        const selectedOption = this.voiceData.voice_options.find(item => item.id === this.selectedVoice);
        const payload = {
            event_id: `voice_demo_${Date.now()}`,
            source_event_id: this.voiceData.crying_event.event_id,
            baby_id: this.voiceData.crying_event.baby_id,
            event_type: 'crying_comfort',
            scene: this.voiceData.crying_event.scene,
            selected_voice: this.selectedVoice,
            selected_voice_label: selectedOption.label,
            comfort_script: document.getElementById('comfortScript').textContent,
            background_audio: this.whiteNoiseToggle.checked ? 'white_noise' : 'none',
            selection_reason: this.selectedVoice === this.voiceData.selected_voice
                ? this.voiceData.selection_reason
                : '用户在 Web Demo 中手动切换了模拟角色音色。',
            is_simulated: true,
            outcome: 'simulated_calmed_after_3min',
        };

        const completeComfortRecord = (logFile) => {
            document.getElementById('comfortResult').textContent =
                'Comfort result: calmed after 3 minutes (simulated demo).';
            document.getElementById('comfortResult').className = 'comfort-result success';
            document.getElementById('comfortLogStatus').textContent =
                `Saved to ${logFile}`;
            document.getElementById('comfortLogStatus').className = 'comfort-log-status success';
            document.querySelector('.playback-panel').classList.add('completed');
        };

        this.recordComfortBtn.disabled = true;
        document.getElementById('comfortLogStatus').textContent = '正在写入模拟安抚记录...';
        try {
            const response = await fetch('/api/voice-companion/result', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const result = await response.json();
            if (!response.ok || !result.ok) throw new Error(result.error || 'Write failed');
            completeComfortRecord(result.log_file);
        } catch (error) {
            const isStaticHost = !['localhost', '127.0.0.1'].includes(window.location.hostname);
            if (isStaticHost) {
                const records = JSON.parse(localStorage.getItem('voiceCompanionDemoRecords') || '[]');
                records.push({ ...payload, timestamp: new Date().toISOString() });
                localStorage.setItem('voiceCompanionDemoRecords', JSON.stringify(records.slice(-20)));
                completeComfortRecord('browser local demo record');
                return;
            }

            console.error('Failed to write comfort result:', error);
            document.getElementById('comfortLogStatus').textContent =
                '写入失败：请使用 web_demo/start_server.py 启动页面';
            document.getElementById('comfortLogStatus').className = 'comfort-log-status error';
            this.recordComfortBtn.disabled = false;
        }
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new BabyMonitorDemo();
    new CryingSimulationDemo();
});

class CryingSimulationDemo {
    constructor() {
        this.runButton = document.getElementById('runCrySimulationBtn');
        this.resetButton = document.getElementById('resetCrySimulationBtn');
        this.select = document.getElementById('cryScenarioSelect');
        if (!this.runButton || !this.resetButton || !this.select) return;
        this.timers = [];
        this.scenarios = this.buildScenarios();
        this.bindEvents();
        this.reset();
    }

    buildScenarios() {
        const memory = {
            count: 6,
            rate: 82,
            strategy: '妈妈音色 + 轻柔摇篮曲（35% 音量）',
            reason: '夜间入睡阶段检测到揉眼；6 次相似历史中，该方案成功率最高。',
            script: '宝宝乖，我在这里呢。慢慢放松，我们准备睡觉。',
        };
        return {
            level1: {
                input: [48, '断续哼唧', '轻微皱眉', '平稳'],
                level: '哭闹 1 级｜轻微呼叫',
                confidence: 88,
                evidence: ['声音处于 40–55dB 区间', '断续哼唧，未形成持续大哭', '体征平稳，无剧烈挣扎'],
                needs: [['陪伴', 58], ['困倦', 30], ['饥饿', 12]],
                memory: { ...memory, rate: 76, strategy: '妈妈安抚语（音量低于 40dB）', reason: '轻度哭闹优先采用低干扰安抚，并静默通知家长观察。' },
                curve: [48, 46, 43, 40, 36],
                result: 'success', state: '1级 → 平静', time: '18 秒', notice: '静默提醒',
                timeline: ['接收仿真传感器事件', '规则判断为哭闹 1 级', '检索轻度呼叫安抚偏好', '创建妈妈音色播放任务', '分贝降至 36dB，判定恢复平静'],
            },
            level2: {
                input: [63, '哇—停—哇', '揉眼、蹬腿', '中度增加'],
                level: '哭闹 2 级｜明确需求',
                confidence: 91,
                evidence: ['声音处于 55–70dB 区间', '呈有节奏的呼唤模式', '检测到揉眼和蹬腿，活动量中度增加'],
                needs: [['困倦', 68], ['陪伴', 22], ['饥饿', 10]],
                memory,
                curve: [63, 59, 54, 49, 43],
                result: 'success', state: '2级 → 1级 → 平静', time: '24 秒', notice: '标准推送',
                timeline: ['接收仿真传感器事件', '规则判断为哭闹 2 级', '推断困倦需求概率为 68%', '命中 6 次相似安抚记忆', '创建妈妈克隆音色播放任务', '哭声降至 54dB，继续观察', '哭声降至 43dB，判定安抚有效'],
            },
            success: {
                input: [66, '持续节奏性哭声', '揉眼、寻求陪伴', '中度增加'],
                level: '哭闹 2 级｜困倦哭闹',
                confidence: 93,
                evidence: ['声音峰值 66dB，属于 2 级区间', '哭声具有稳定节奏', '夜间入睡场景伴随揉眼动作'],
                needs: [['困倦', 74], ['陪伴', 19], ['饥饿', 7]],
                memory: { ...memory, rate: 84 },
                curve: [66, 58, 51, 45, 38],
                result: 'success', state: '2级 → 平静', time: '22 秒', notice: '已通知，无需升级',
                timeline: ['接收仿真哭闹事件', '完成多源规则匹配', '历史记忆推荐妈妈音色', '创建安抚播放任务', '哭声下降 8dB，继续执行', '哭声降至 38dB，闭环成功', '生成新的安抚效果记忆'],
            },
            escalation: {
                input: [64, '节奏性哭声转尖锐', '蹬腿、持续挣扎', '快速上升'],
                level: '哭闹 2 级｜存在升级趋势',
                confidence: 90,
                evidence: ['初始声音处于 55–70dB 区间', '活动量持续上升', '测试序列将在 30 秒后超过 75dB'],
                needs: [['不适', 52], ['陪伴', 28], ['困倦', 20]],
                memory,
                curve: [64, 67, 71, 76, 79],
                result: 'danger', state: '2级 → 3级告警', time: '30 秒', notice: '强提醒家长',
                timeline: ['接收仿真传感器事件', '初始判断为哭闹 2 级', '执行历史最优安抚方案', '哭声未下降，活动量继续上升', '声音升至 79dB，升级为 3 级', '停止背景音乐并切换父母安慰音', '发送强震动告警并记录时间'],
            },
        };
    }

    bindEvents() {
        this.runButton.addEventListener('click', () => this.run());
        this.resetButton.addEventListener('click', () => this.reset());
        document.querySelectorAll('[data-sim-voice]').forEach(button => button.addEventListener('click', () => {
            document.querySelectorAll('[data-sim-voice]').forEach(item => item.classList.toggle('selected', item === button));
            const role = button.dataset.simVoice === 'mother' ? '妈妈' : '爸爸';
            document.getElementById('simComfortScript').textContent = `${role}音色已设为人工接管；真实 voice_id 接入后将替换当前播放任务。`;
        }));
        document.querySelectorAll('[data-feedback]').forEach(button => button.addEventListener('click', () => {
            document.getElementById('simMemoryUpdate').textContent = button.dataset.feedback === 'accurate'
                ? '已接收人工反馈：判断准确（本地仿真，不写入生产数据库）。'
                : '已标记为需要人工纠正（本地仿真，不写入生产数据库）。';
        }));
    }

    _playSyntheticLullaby(vol) {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            this._lullabyCtx = ctx;
            const notes = [
                { f: 261.6, d: 0.4 }, { f: 261.6, d: 0.4 }, { f: 392.0, d: 0.4 }, { f: 392.0, d: 0.4 },
                { f: 440.0, d: 0.4 }, { f: 440.0, d: 0.4 }, { f: 392.0, d: 0.8 },
                { f: 349.2, d: 0.4 }, { f: 349.2, d: 0.4 }, { f: 329.6, d: 0.4 }, { f: 329.6, d: 0.4 },
                { f: 293.7, d: 0.4 }, { f: 293.7, d: 0.4 }, { f: 261.6, d: 0.8 },
            ];
            let t = ctx.currentTime + 0.05;
            for (let i = 0; i < 3; i++) {
                notes.forEach(n => {
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.type = 'triangle';
                    osc.frequency.value = n.f;
                    gain.gain.setValueAtTime(0, t);
                    gain.gain.linearRampToValueAtTime(vol * 0.18, t + 0.08);
                    gain.gain.linearRampToValueAtTime(vol * 0.14, t + n.d * 0.6);
                    gain.gain.linearRampToValueAtTime(0, t + n.d);
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.start(t);
                    osc.stop(t + n.d + 0.01);
                    t += n.d;
                });
                t += 0.3;
            }
        } catch (e) { /* Web Audio API 不支持则跳过 */ }
    }

    clearTimers() {
        this.timers.forEach(id => clearTimeout(id));
        this.timers = [];
    }

    reset() {
        this.clearTimers();
        this.runButton.disabled = false;
        this.select.disabled = false;
        const values = {
            simDecibel: '-- dB', simRhythm: '--', simMotion: '--', simActivity: '--',
            simCryLevel: '等待测试数据', simConfidence: '--', simStateChange: '--',
            simResponseTime: '--', simNotification: '--',
        };
        Object.entries(values).forEach(([id, value]) => { document.getElementById(id).textContent = value; });
        const badge = document.getElementById('simulationStateBadge');
        badge.textContent = '等待输入';
        badge.className = '';
        document.getElementById('simEvidenceList').innerHTML = '<li>等待注入测试场景</li>';
        document.getElementById('simNeedBars').innerHTML = '';
        document.getElementById('simDecisionEmpty').classList.remove('hidden');
        document.getElementById('simDecisionContent').classList.add('hidden');
        document.getElementById('simTimeline').innerHTML = '<li><time>--:--</time><span>等待执行</span></li>';
        document.getElementById('simOutcome').textContent = '等待系统自动评估';
        document.getElementById('simOutcome').className = 'comfort-result pending';
        const memoryUpdate = document.getElementById('simMemoryUpdate');
        memoryUpdate.textContent = '尚未生成新的安抚记忆';
        memoryUpdate.className = 'memory-update';
        document.querySelectorAll('#cryingSimulationSection .simulation-panel').forEach(panel => panel.classList.remove('running', 'completed', 'alert'));
        document.querySelectorAll('#simChartBars span').forEach(bar => { bar.style.height = '8px'; });
        document.querySelectorAll('[data-feedback]').forEach(button => { button.disabled = true; });
        // 监控小窗重置
        const monitor = document.getElementById('simMonitorWindow');
        if (monitor) monitor.classList.add('hidden');
        const monitorMsg = document.getElementById('monitorMessage');
        if (monitorMsg) { monitorMsg.textContent = '宝宝可能饿了 / 要抱抱'; monitorMsg.classList.remove('calm'); }
        const monitorTime = document.getElementById('monitorTime');
        if (monitorTime) monitorTime.textContent = '00:00';
        // 停止所有音频
        [this._simCrying, this._simLullaby, this._simComfort].forEach(a => {
            if (a) { try { a.pause(); a.currentTime = 0; } catch(e) {} }
        });
        this._simCrying = null;
        this._simLullaby = null;
        this._simComfort = null;
        if (this._lullabyCtx) { try { this._lullabyCtx.close(); } catch(e) {} this._lullabyCtx = null; }
        if (this._fadeOutInterval) { clearInterval(this._fadeOutInterval); this._fadeOutInterval = null; }
        if (this._monitorTimer) { clearInterval(this._monitorTimer); this._monitorTimer = null; }
    }

    renderNeeds(needs) {
        document.getElementById('simNeedBars').innerHTML = needs.map(([label, value]) =>
            `<div class="need-row"><span>${label}</span><div class="need-track"><i style="width:${value}%"></i></div><strong>${value}%</strong></div>`
        ).join('');
    }

    run() {
        this.reset();
        const scenario = this.scenarios[this.select.value];
        this.runButton.disabled = true;
        this.select.disabled = true;

        const badge = document.getElementById('simulationStateBadge');
        const later = (delay, fn) => this.timers.push(setTimeout(fn, delay));

        // ── T=0s：开始播放哭声 ──
        badge.textContent = '检测中';
        badge.className = 'running';
        document.querySelector('.perception-panel').classList.add('running');

        this._simCrying = new Audio(CRYING_TEST_AUDIO);
        this._simCrying.volume = 0.7;
        this._simCrying.loop = true;
        this._simCrying.play()
            .then(() => {
                const statusEl = document.getElementById('cryingSlotStatus');
                if (statusEl) statusEl.textContent = '婴儿哭声播放中，正在检测声强阈值';
            })
            .catch(() => {
                const statusEl = document.getElementById('cryingSlotStatus');
                if (statusEl) statusEl.textContent = '浏览器阻止自动播放，请再次点击注入测试场景';
            });
        const statusEl = document.getElementById('cryingSlotStatus');
        if (statusEl) statusEl.textContent = '播放 442655__josephvm__baby-girl-crying.wav，等待哭声阈值';
        this._simCrying.onended = () => { if (statusEl) statusEl.textContent = '播放结束'; };

        // ── T=1s：感知数据填充 ──
        later(1000, () => {
            ['simDecibel', 'simRhythm', 'simMotion', 'simActivity'].forEach((id, index) => {
                document.getElementById(id).textContent = index === 0 ? `${scenario.input[index]} dB` : scenario.input[index];
            });
        });

        // ── T=2s：分类完成，badge="哭闹 2 级" ──
        later(2000, () => {
            document.getElementById('simCryLevel').textContent = scenario.level;
            document.getElementById('simConfidence').textContent = `${scenario.confidence}%`;
            document.getElementById('simEvidenceList').innerHTML = scenario.evidence.map(item => `<li>${item}</li>`).join('');
            this.renderNeeds(scenario.needs);
            document.querySelector('.perception-panel').className = 'simulation-panel perception-panel completed';
            const thresholdReached = scenario.input[0] >= CRYING_TRIGGER_THRESHOLD_DB;
            badge.textContent = thresholdReached ? `${scenario.level} / 阈值已触发` : scenario.level;
            badge.className = 'success';
            if (statusEl) {
                statusEl.textContent = thresholdReached
                    ? `哭声 ${scenario.input[0]}dB >= ${CRYING_TRIGGER_THRESHOLD_DB}dB，触发安抚决策`
                    : `哭声 ${scenario.input[0]}dB，未达到强安抚阈值，继续观察`;
            }
        });

        // ── T=3s：监控小窗弹出，继续监听哭声 ──
        later(3000, () => {
            const monitor = document.getElementById('simMonitorWindow');
            const monitorMsg = document.getElementById('monitorMessage');
            const monitorTimeEl = document.getElementById('monitorTime');
            if (monitor) { monitor.classList.remove('hidden'); monitor.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
            if (monitorMsg) { monitorMsg.textContent = '宝宝可能饿了 / 要抱抱'; monitorMsg.classList.remove('calm'); }
            let _sec = 0;
            this._monitorTimer = setInterval(() => {
                _sec++;
                const mm = String(Math.floor(_sec / 60)).padStart(2, '0');
                const ss = String(_sec % 60).padStart(2, '0');
                if (monitorTimeEl) monitorTimeEl.textContent = `${mm}:${ss}`;
            }, 1000);

            if (this._simCrying) this._simCrying.volume = 0.65;
        });

        // ── T=5s：决策面板填充 ──
        later(5000, () => {
            document.getElementById('simDecisionEmpty').classList.add('hidden');
            document.getElementById('simDecisionContent').classList.remove('hidden');
            document.getElementById('simStrategy').textContent = scenario.memory.strategy;
            document.getElementById('simStrategyReason').textContent = scenario.memory.reason;
            document.getElementById('simHistoryCount').textContent = `${scenario.memory.count} 次`;
            document.getElementById('simSuccessRate').textContent = `${scenario.memory.rate}%`;
            document.getElementById('simComfortScript').textContent = scenario.memory.script;
            document.querySelector('.decision-panel').classList.add('completed');
        });

        // ── T=7s：timeline 逐步执行 ──
        later(7000, () => {
            document.querySelector('.execution-panel').classList.add('running');
            document.getElementById('simTimeline').innerHTML = scenario.timeline.map((text, index) =>
                `<li><time>00:${String(index * 5).padStart(2, '0')}</time><span>${text}</span></li>`
            ).join('');
            const items = [...document.querySelectorAll('#simTimeline li')];
            items.forEach((item, index) => later(index * 800, () => {
                items.forEach((entry, entryIndex) => {
                    if (entryIndex < index) entry.className = 'done';
                    if (entryIndex === index) entry.className = scenario.result === 'danger' && index >= items.length - 3 ? 'danger' : 'active';
                });
                const bar = document.querySelectorAll('#simChartBars span')[Math.min(index, 4)];
                if (bar) bar.style.height = `${Math.max(10, ((scenario.curve[Math.min(index, 4)] - 30) / 50) * 100)}%`;
            }));
        });

        // ── T=8.5s：阈值触发后播放父母音色安抚与背景摇篮曲 ──
        later(8500, () => {
            if (statusEl) statusEl.textContent = '阈值触发后开始父母音色安抚，哭声逐步降低';
            this._playSyntheticLullaby(0.35);
            this._simComfort = new Audio('assets/audio/parent_comfort.wav');
            this._simComfort.volume = 0.8;
            this._simComfort.play().catch(() => {});
            if (this._simCrying) this._simCrying.volume = 0.3;
        });

        // ── T=20s：哭声渐弱 0.3 → 0.1 ──
        later(20000, () => {
            if (this._simCrying && !this._simCrying.paused) {
                let vol = 0.3;
                this._fadeOutInterval = setInterval(() => {
                    vol = Math.max(0.1, vol - 0.04);
                    if (this._simCrying) this._simCrying.volume = vol;
                    if (vol <= 0.1) {
                        clearInterval(this._fadeOutInterval);
                        this._fadeOutInterval = null;
                    }
                }, 500);
            }
        });

        // ── T=25s：哭声停止，监控小窗平静 ──
        later(25000, () => {
            if (this._simCrying) { this._simCrying.pause(); this._simCrying.currentTime = 0; }
            if (this._fadeOutInterval) { clearInterval(this._fadeOutInterval); this._fadeOutInterval = null; }
            if (statusEl) statusEl.textContent = '哭闹已停止 — 安抚闭环完成';

            const monitorMsg = document.getElementById('monitorMessage');
            if (monitorMsg) { monitorMsg.textContent = '宝宝已平静 💤'; monitorMsg.classList.add('calm'); }

            document.querySelector('.execution-panel').className = 'simulation-panel execution-panel completed';
            badge.textContent = '闭环完成';
            badge.className = 'success';

            this.finish(scenario);
            this.runButton.disabled = false;
            this.select.disabled = false;
            document.querySelectorAll('[data-feedback]').forEach(button => { button.disabled = false; });
        });
    }

    finish(scenario) {
        document.querySelector('.execution-panel').className = 'simulation-panel execution-panel completed';
        document.querySelectorAll('#simChartBars span').forEach((bar, index) => {
            bar.style.height = `${Math.max(10, ((scenario.curve[index] - 30) / 50) * 100)}%`;
            bar.title = `${scenario.curve[index]}dB`;
        });
        document.getElementById('simStateChange').textContent = scenario.state;
        document.getElementById('simResponseTime').textContent = scenario.time;
        document.getElementById('simNotification').textContent = scenario.notice;
        const outcome = document.getElementById('simOutcome');
        const memoryUpdate = document.getElementById('simMemoryUpdate');
        const badge = document.getElementById('simulationStateBadge');
        if (scenario.result === 'danger') {
            outcome.textContent = '安抚未生效：已自动升级为哭闹 3 级告警';
            outcome.className = 'comfort-result warning';
            memoryUpdate.textContent = '已生成失败样本：下次降低当前策略权重，并优先要求家长介入。';
            memoryUpdate.className = 'memory-update danger';
            document.querySelector('.outcome-panel').classList.add('alert');
            badge.textContent = '已升级告警';
            badge.className = 'danger';
        } else {
            outcome.textContent = `安抚有效：哭声由 ${scenario.curve[0]}dB 降至 ${scenario.curve.at(-1)}dB`;
            outcome.className = 'comfort-result success';
            memoryUpdate.textContent = `已生成仿真记忆：${scenario.memory.strategy} 有效；成功率 ${scenario.memory.rate}% → ${scenario.memory.rate + 2}%。`;
            memoryUpdate.className = 'memory-update success';
            document.querySelector('.outcome-panel').classList.add('completed');
            badge.textContent = '闭环完成';
            badge.className = 'success';
        }
        document.querySelectorAll('[data-feedback]').forEach(button => { button.disabled = false; });
        this.runButton.disabled = false;
        this.select.disabled = false;
    }
}
