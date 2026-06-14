/**
 * AI婴儿床监护系统 - 网页端Demo
 * YOLO危险动作检测可视化
 * 
 * 使用预计算的YOLO检测数据，检测框精确跟随婴儿身体移动
 */

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

        // 数据存储
        this.detectionData = null;  // 预计算的YOLO检测数据
        this.framesByIndex = null;  // 按帧索引索引的Map，加速查找
        this.videoInfo = null;      // 视频信息（分辨率、fps等）
        this.geometry = null;       // 几何配置（安全区、警告区、危险边界）
        this.currentVideo = 'dangerous_test1';
        this.events = [];
        this.triggeredEventKeys = new Set();

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
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

        // 加载视频
        this.videoPlayer.src = `data/${videoName}.mp4`;

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
     * 坐标从视频原始分辨率映射到Canvas显示尺寸
     */
    drawZones(scaleX, scaleY) {
        const geo = this.geometry;
        if (!geo) return;

        const [sx, sy, sw, sh] = geo.safe_zone;
        const [wx, wy, ww, wh] = geo.warning_zone;
        const dangerX = geo.danger_boundary_x;

        // 映射到Canvas坐标
        const safeX = sx * scaleX, safeY = sy * scaleY, safeW = sw * scaleX, safeH = sh * scaleY;
        const warnX = wx * scaleX, warnY = wy * scaleY, warnW = ww * scaleX, warnH = wh * scaleY;
        const dangerXCanvas = dangerX * scaleX;

        // 安全区（绿色半透明）
        this.ctx.strokeStyle = 'rgba(74, 222, 128, 0.8)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([8, 4]);
        this.ctx.strokeRect(safeX, safeY, safeW, safeH);
        this.ctx.setLineDash([]);
        this.ctx.fillStyle = 'rgba(74, 222, 128, 0.08)';
        this.ctx.fillRect(safeX, safeY, safeW, safeH);

        // 安全区标签
        this.ctx.fillStyle = 'rgba(74, 222, 128, 0.9)';
        this.ctx.font = `${Math.max(12, 14 * scaleX)}px sans-serif`;
        this.ctx.fillText('SAFE ZONE', safeX + 8, safeY + 20 * scaleY);

        // 警告区（黄色半透明）
        this.ctx.strokeStyle = 'rgba(251, 191, 36, 0.8)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([8, 4]);
        this.ctx.strokeRect(warnX, warnY, warnW, warnH);
        this.ctx.setLineDash([]);
        this.ctx.fillStyle = 'rgba(251, 191, 36, 0.08)';
        this.ctx.fillRect(warnX, warnY, warnW, warnH);

        // 警告区标签
        this.ctx.fillStyle = 'rgba(251, 191, 36, 0.9)';
        this.ctx.fillText('WARNING', warnX + 8, warnY + 20 * scaleY);

        // 危险边界线（红色虚线）
        this.ctx.strokeStyle = 'rgba(248, 113, 113, 0.9)';
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([10, 5]);
        this.ctx.beginPath();
        this.ctx.moveTo(dangerXCanvas, 0);
        this.ctx.lineTo(dangerXCanvas, this.canvas.height);
        this.ctx.stroke();
        this.ctx.setLineDash([]);

        // 危险边界标签
        this.ctx.fillStyle = 'rgba(248, 113, 113, 0.9)';
        this.ctx.font = `bold ${Math.max(12, 14 * scaleX)}px sans-serif`;
        this.ctx.fillText('DANGER BOUNDARY', dangerXCanvas - 100 * scaleX, 25 * scaleY);
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
        const label = `${target.class_name || 'person'}  ${(target.confidence * 100).toFixed(1)}%`;
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

            // bbox_right 标签
            this.ctx.fillStyle = color;
            this.ctx.font = `${Math.max(10, 11 * scaleX)}px sans-serif`;
            this.ctx.fillText(`bbox_right=${Math.round(x + w)}`, bboxRight + 5, by + bh / 2);
        }
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
        } else if (stage === 'WARNING') {
            this.statusOverlay.classList.add('warning');
            this.dangerOverlay.classList.add('hidden');
        } else if (stage === 'DANGEROUS_ACTION') {
            this.statusOverlay.classList.add('danger');
            this.dangerOverlay.classList.remove('hidden');
        }
        this.statusBadge.textContent = stage;

        // 更新右侧面板
        this.currentStageEl.textContent = stage;
        this.currentStageEl.className = 'value';
        if (stage === 'SAFE') this.currentStageEl.classList.add('stage-safe');
        else if (stage === 'WARNING') this.currentStageEl.classList.add('stage-warning');
        else if (stage === 'DANGEROUS_ACTION') this.currentStageEl.classList.add('stage-danger');

        this.rawStageEl.textContent = rawStage;

        if (frameData.target) {
            this.confidenceEl.textContent = frameData.target.confidence.toFixed(4);
            this.targetPosEl.textContent = `(${Math.round(frameData.target.center.x)}, ${Math.round(frameData.target.center.y)})`;
        } else {
            this.confidenceEl.textContent = '--';
            this.targetPosEl.textContent = '--';
        }

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
            // 使用事件数据中的帧索引作为key（和预计算脚本一致）
            const eventKey = String(frameData.frame_index);

            if (!this.triggeredEventKeys.has(eventKey)) {
                // 检查预计算事件列表中是否有匹配的事件
                const matchedEvent = this.detectionData.events.find(
                    e => Math.abs(e.frame_index - frameData.frame_index) < 5
                );

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
                }
            }
        }
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
                    危险动作触发 | 置信度: ${(event.confidence * 100).toFixed(1)}% |
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
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new BabyMonitorDemo();
});
