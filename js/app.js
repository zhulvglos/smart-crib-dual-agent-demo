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

        // 数据存储
        this.detectionData = null;  // 预计算的YOLO检测数据
        this.framesByIndex = null;  // 按帧索引索引的Map，加速查找
        this.videoInfo = null;      // 视频信息（分辨率、fps等）
        this.geometry = null;       // 几何配置（安全区、警告区、危险边界）
        this.currentVideo = 'dangerous_test6';
        this.events = [];
        this.triggeredEventKeys = new Set();

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        this.loadGrowthMemory();
        this.videoSelect.value = this.currentVideo;
        this.loadVideo(this.currentVideo);
        // 预加载语音列表（部分浏览器异步加载）
        if ('speechSynthesis' in window) {
            window.speechSynthesis.getVoices();
            window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
        }
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
     * Web Speech API 语音播报（真实执行，非模拟）
     */
    speakWarning() {
        if (!('speechSynthesis' in window)) return;
        // 避免重复播报
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance('宝贝小心，请往中间来');
        utterance.lang = 'zh-CN';
        utterance.rate = 1.0;
        utterance.pitch = 1.1;
        utterance.volume = 0.9;
        // 优先选择中文语音
        const voices = window.speechSynthesis.getVoices();
        const zhVoice = voices.find(v => v.lang.startsWith('zh'));
        if (zhVoice) utterance.voice = zhVoice;
        window.speechSynthesis.speak(utterance);
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
            </div>
        `).join('');
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new BabyMonitorDemo();
});
