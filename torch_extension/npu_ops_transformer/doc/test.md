<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
    <title>产品参数对比 · 纯CSS选项卡表格</title>
    <style>
        /* 完全使用内部样式 —— 无外部资源，无JS，兼容所有现代浏览器与VSCode Live Preview */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            background: #eef2f8;
            font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 2rem 1rem;
        }
        /* 主容器：所有radio、label和表格区域都在此容器内，无额外包裹层，确保兄弟选择器有效 */
        .product-comparator {
            max-width: 1000px;
            width: 100%;
            background: #ffffff;
            border-radius: 1.8rem;
            box-shadow: 0 12px 30px rgba(0,0,0,0.08);
            overflow: hidden;
            transition: all 0.2s;
        }
        /* 头部信息 */
        .comparator-header {
            padding: 1.5rem 2rem 0.8rem 2rem;
            border-bottom: 1px solid #eef2f8;
            background: #fefefe;
        }
        .comparator-header h1 {
            font-size: 1.6rem;
            font-weight: 600;
            color: #1f4e6e;
            letter-spacing: -0.2px;
        }
        .comparator-header p {
            color: #5f7f9c;
            font-size: 0.9rem;
            margin-top: 0.4rem;
        }
        /* ---------- 核心：隐藏原生radio，用label模拟按钮 ---------- */
        /* 所有radio不显示，但保留DOM以便使用 :checked 伪类 */
        .tab-radio {
            display: none;
        }
        /* 按钮样式：与radio标签紧邻的label（每个radio后面跟着一个label） */
        .tab-label {
            display: inline-block;
            background: #f0f4fa;
            padding: 0.65rem 1.8rem;
            margin: 0.4rem 0.6rem 0.4rem 0;
            border-radius: 60px;
            font-size: 0.95rem;
            font-weight: 500;
            color: #2c4c6e;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid #dee6ef;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            text-align: center;
        }
        .tab-label:hover {
            background: #e6edf6;
            transform: translateY(-1px);
            border-color: #c0d4e4;
        }
        /* 选中按钮的高亮状态：通过 radio:checked + label 实现（因为label是紧邻的后续兄弟） */
        .tab-radio:checked + .tab-label {
            background: #1f6e9c;
            color: white;
            border-color: #1f6e9c;
            box-shadow: 0 6px 14px rgba(31,110,156,0.25);
        }
        /* ---------- 表格区域容器 ---------- */
        .tables-container {
            padding: 1.2rem 2rem 2rem 2rem;
            background: #ffffff;
        }
        /* 每个产品的表格面板，默认隐藏 */
        .product-panel {
            display: none;
            width: 100%;
            overflow-x: auto;
            animation: fadeIn 0.25s ease;
        }
        /* 关键：使用通用兄弟选择器 ~ 控制显示哪个面板 */
        /* 注意结构：所有 radio 和 .tables-container 是兄弟关系，且 .tables-container 内部的面板通过 radio:checked 来选择显示 */
        /* 因为 radio 与 .tables-container 同级，所以选择器 #tab1:checked ~ .tables-container .panel-1 有效 */
        #tab1:checked ~ .tables-container .panel-1,
        #tab2:checked ~ .tables-container .panel-2,
        #tab3:checked ~ .tables-container .panel-3 {
            display: block;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }
        /* 表格样式：清爽的Markdown风格表格，干净易读 */
        .spec-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
            background: white;
            border-radius: 1rem;
            overflow: hidden;
            border: 1px solid #e4ebf3;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }
        .spec-table th,
        .spec-table td {
            padding: 1rem 1.2rem;
            text-align: left;
            border-bottom: 1px solid #ecf3f9;
        }
        .spec-table th {
            background-color: #fafdff;
            font-weight: 600;
            color: #1d4d6f;
            width: 34%;
            font-size: 0.88rem;
            letter-spacing: 0.2px;
            border-right: 1px solid #eef3fa;
        }
        .spec-table td {
            background-color: #ffffff;
            color: #253c55;
        }
        .spec-table tr:last-child th,
        .spec-table tr:last-child td {
            border-bottom: none;
        }
        /* 价格特殊标记 */
        .price-badge {
            font-weight: 700;
            color: #c1432e;
            background: #fff0ec;
            display: inline-block;
            padding: 0.2rem 0.9rem;
            border-radius: 30px;
            font-size: 0.85rem;
        }
        /* 底部提示 */
        .footer-note {
            padding: 0.8rem 2rem 1.3rem 2rem;
            font-size: 0.7rem;
            color: #8da0b5;
            text-align: center;
            border-top: 1px solid #ecf2f8;
            background: #fdfdfd;
        }
        /* 响应式微调 */
        @media (max-width: 650px) {
            .comparator-header {
                padding: 1.2rem 1.2rem 0.5rem 1.2rem;
            }
            .tables-container {
                padding: 1rem 1.2rem 1.5rem 1.2rem;
            }
            .spec-table th,
            .spec-table td {
                padding: 0.75rem 0.9rem;
            }
            .tab-label {
                padding: 0.5rem 1.2rem;
                margin: 0.3rem 0.4rem;
                font-size: 0.85rem;
            }
        }
    </style>
</head>
<body>
<div class="product-comparator">
    <div class="comparator-header">
        <h1>📊 产品参数对比</h1>
        <p>点击下方选项按钮，即时切换查看不同产品的完整规格表（纯CSS实现，无JavaScript）</p>
    </div>
    <!-- 
        注意：所有radio和表格容器必须为兄弟节点，且radio隐藏。 
        每个radio后面紧跟对应的label作为按钮，保证 :checked + label 选中样式有效。
        多个radio使用相同的name确保互斥性。
    -->
    <input type="radio" name="product_group" id="tab1" class="tab-radio" checked>
    <label for="tab1" class="tab-label">⌚ 智能手表 X9</label>
    <input type="radio" name="product_group" id="tab2" class="tab-radio">
    <label for="tab2" class="tab-label">🏃 运动手环 Fit3</label>
    <input type="radio" name="product_group" id="tab3" class="tab-radio">
    <label for="tab3" class="tab-label">⛰️ 户外手表 Runner Pro</label>
    <!-- 表格容器，包含三个产品面板，通过 radio 控制显隐 -->
    <div class="tables-container">
        <!-- 产品1：智能手表 X9 的参数表格 -->
        <div class="product-panel panel-1">
            <table class="spec-table">
                <thead>
                    <tr><th>📌 参数项目</th><th>⚙️ 规格详情</th></tr>
                </thead>
                <tbody>
                    <tr><th>🖥️ 屏幕尺寸</th><td>1.8 英寸 AMOLED, 常亮显示 (1000尼特)</td></tr>
                    <tr><th>🔋 电池续航</th><td>7 天 (典型使用) / 15 天省电模式</td></tr>
                    <tr><th>💧 防水等级</th><td>5 ATM (50米防水，适合游泳)</td></tr>
                    <tr><th>❤️ 心率监测</th><td>全天候心率监测 + 血氧饱和度 (SpO2)</td></tr>
                    <tr><th>📍 GPS 定位</th><td>独立GPS / GLONASS / 北斗三模</td></tr>
                    <tr><th>✨ 特色功能</th><td>蓝牙通话 / 离线支付 / 30+运动模式 / 通知提醒</td></tr>
                    <tr><th>💰 参考价格</th><td><span class="price-badge">¥1,299</span></td></tr>
                </tbody>
            </table>
        </div>
        <!-- 产品2：运动手环 Fit3 的参数表格 -->
        <div class="product-panel panel-2">
            <table class="spec-table">
                <thead><tr><th>📌 参数项目</th><th>⚙️ 规格详情</th></tr></thead>
                <tbody>
                    <tr><th>🖥️ 屏幕尺寸</th><td>1.2 英寸 OLED, 触摸屏，2.5D玻璃</td></tr>
                    <tr><th>🔋 电池续航</th><td>14 天超长续航 (磁吸快充)</td></tr>
                    <tr><th>💧 防水等级</th><td>3 ATM (30米防水，支持浅滩游泳)</td></tr>
                    <tr><th>❤️ 心率监测</th><td>智能心率监测 + 深度睡眠分析</td></tr>
                    <tr><th>📍 GPS 定位</th><td>不支持独立GPS (连接手机GPS使用)</td></tr>
                    <tr><th>✨ 特色功能</th><td>消息推送 / 久坐提醒 / 女性生理周期 / 遥控拍照</td></tr>
                    <tr><th>💰 参考价格</th><td><span class="price-badge">¥399</span></td></tr>
                </tbody>
            </table>
        </div>
        <!-- 产品3：户外手表 Runner Pro 的参数表格 -->
        <div class="product-panel panel-3">
            <table class="spec-table">
                <thead><tr><th>📌 参数项目</th><th>⚙️ 规格详情</th></tr></thead>
                <tbody>
                    <tr><th>🖥️ 屏幕尺寸</th><td>1.4 英寸 半透反射式屏幕 (阳光下清晰可见)</td></tr>
                    <tr><th>🔋 电池续航</th><td>5 天 (GPS全开) / 20天 纯手表模式</td></tr>
                    <tr><th>💧 防水等级</th><td>10 ATM (100米防水，自由潜/游泳可用)</td></tr>
                    <tr><th>❤️ 心率监测</th><td>4通道心率传感器 + 心率变异分析(HRV)</td></tr>
                    <tr><th>📍 GPS 定位</th><td>双频多星定位 (GPS/北斗/Galileo/GLONASS)</td></tr>
                    <tr><th>✨ 特色功能</th><td>离线地图导航 / 气压高度计 / 轨迹返航 / 血氧监测</td></tr>
                    <tr><th>💰 参考价格</th><td><span class="price-badge">¥2,199</span></td></tr>
                </tbody>
            </table>
        </div>
    </div>
    <div class="footer-note">
        ⚡ 采用纯HTML + CSS 选项卡技术（radio + 兄弟选择器）| 点击按钮即时切换 | 无需任何 JavaScript | 完全兼容 VSCode 及主流 Markdown 环境
    </div>
</div>
</body>
</html>