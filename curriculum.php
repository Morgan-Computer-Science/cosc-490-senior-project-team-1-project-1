<?php
session_start();

if(!isset($_SESSION["user_id"])){
    header("Location: index.php");
    exit();
}

$name = $_SESSION["name"] ?? "Student";
?>

<!DOCTYPE html>
<html>

<head>
<title>Morgan AI - Curriculum</title>
<link rel="stylesheet" href="styles.css">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    background: #0d0d0d;
    color: #f0f0f0;
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
}

.curr-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    background: #111;
    border-bottom: 1px solid #222;
    position: sticky;
    top: 0;
    z-index: 100;
}

.curr-header .logo {
    height: 40px;
    width: auto;
    cursor: pointer;
}

.curr-header .header-right {
    display: flex;
    align-items: center;
    gap: 16px;
}

.curr-header .header-right .icon {
    height: 28px;
    cursor: pointer;
    opacity: 0.8;
    transition: opacity .2s;
}

.curr-header .header-right .icon:hover {
    opacity: 1;
}

.back-btn {
    background: #1e1e1e;
    border: 1px solid #333;
    color: #f0f0f0;
    font-family: 'Inter', sans-serif;
    font-size: .82rem;
    font-weight: 500;
    padding: 7px 16px;
    border-radius: 8px;
    cursor: pointer;
    text-decoration: none;
    transition: background .2s, border-color .2s;
}

.back-btn:hover {
    background: #2a2a2a;
    border-color: #c8102e;
    color: #fff;
}

.curr-hero {
    padding: 48px 32px 32px;
    background: linear-gradient(160deg, #1a0008 0%, #0d0d0d 55%);
    border-bottom: 1px solid #1e1e1e;
}

.curr-hero .label {
    font-size: .72rem;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: #c8102e;
    font-weight: 600;
    margin-bottom: 10px;
}

.curr-hero h1 {
    font-size: clamp(1.6rem, 3.5vw, 2.4rem);
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 10px;
}

.curr-hero h1 span {
    color: #c8102e;
}

.curr-hero p {
    color: #888;
    font-size: .9rem;
    max-width: 520px;
    line-height: 1.6;
}

.filter-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    padding: 24px 32px 0;
}

.filter-btn {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    color: #888;
    font-family: 'Inter', sans-serif;
    font-size: .8rem;
    font-weight: 500;
    padding: 7px 18px;
    border-radius: 999px;
    cursor: pointer;
    transition: all .2s;
}

.filter-btn:hover {
    border-color: #c8102e;
    color: #f0f0f0;
}

.filter-btn.active {
    background: #c8102e;
    border-color: #c8102e;
    color: #fff;
}

.legend-row {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    padding: 20px 32px 0;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: .78rem;
    color: #888;
}

.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

.progress-section {
    margin: 24px 32px 0;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 20px 24px;
}

.progress-section h2 {
    font-size: .95rem;
    font-weight: 600;
    margin-bottom: 16px;
    color: #f0f0f0;
}

.progress-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 11px;
}

.progress-row:last-child {
    margin-bottom: 0;
}

.progress-label {
    font-size: .78rem;
    color: #888;
    min-width: 140px;
}

.progress-track {
    flex: 1;
    height: 6px;
    background: #2a2a2a;
    border-radius: 99px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 1.2s ease;
}

.progress-val {
    font-size: .75rem;
    color: #666;
    min-width: 64px;
    text-align: right;
}

.semester-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 18px;
    padding: 24px 32px 60px;
}

.semester-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    overflow: hidden;
    transition: transform .2s, border-color .2s, box-shadow .2s;
}

.semester-card:hover {
    transform: translateY(-3px);
    border-color: #c8102e;
    box-shadow: 0 6px 24px rgba(200,16,46,.12);
}

.card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 18px 12px;
    border-bottom: 1px solid #2a2a2a;
}

.card-head-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.year-badge {
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 6px;
    color: #fff;
}

.y1 { background: #c8102e; }
.y2 { background: #2563eb; }
.y3 { background: #059669; }
.y4 { background: #d97706; }

.card-head h3 {
    font-size: .9rem;
    font-weight: 600;
    color: #f0f0f0;
}

.credit-badge {
    font-size: .72rem;
    color: #666;
    background: rgba(255,255,255,.05);
    padding: 3px 10px;
    border-radius: 20px;
}

.course-list {
    padding: 12px 18px 16px;
    display: flex;
    flex-direction: column;
    gap: 7px;
}

.course-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 9px 11px;
    border-radius: 8px;
    background: rgba(255,255,255,.02);
    border: 1px solid transparent;
    transition: background .15s, border-color .15s;
}

.course-item:hover {
    background: rgba(200,16,46,.07);
    border-color: rgba(200,16,46,.18);
}

.course-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}

.dot-core     { background: #c8102e; }
.dot-support  { background: #10b981; }
.dot-gen      { background: #6366f1; }
.dot-elective { background: #f59e0b; }

.course-info {
    flex: 1;
}

.course-code {
    font-size: .67rem;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #555;
}

.course-name {
    font-size: .85rem;
    font-weight: 400;
    color: #d0d0d0;
    margin-top: 1px;
    line-height: 1.35;
}

.course-credits {
    font-size: .7rem;
    color: #555;
    white-space: nowrap;
    padding-top: 3px;
}

@media (max-width: 600px) {
    .curr-hero,
    .filter-row,
    .legend-row,
    .semester-grid {
        padding-left: 16px;
        padding-right: 16px;
    }
    .progress-section {
        margin-left: 16px;
        margin-right: 16px;
    }
    .semester-grid {
        grid-template-columns: 1fr;
    }
}

</style>
</head>

<body>

<div class="curr-header">

    <a href="dashboard.php">
        <img src="assets/logo.png" class="logo" alt="Morgan AI">
    </a>

    <div class="header-right">
        <a href="dashboard.php" class="back-btn">← Back to Chat</a>
        <a href="settings.php">
            <img src="assets/user_icon.png" class="icon" alt="Settings">
        </a>
    </div>

</div>

<div class="curr-hero">
    <p class="label">Morgan State University · Computer Science</p>
    <h1>Your 4-Year <span>Curriculum</span> Roadmap</h1>
    <p>Suggested course sequence for the BS Computer Science program. Use the filters below to focus on a specific year.</p>
</div>

<div class="filter-row">
    <button class="filter-btn active" onclick="filterCards('all', this)">All Semesters</button>
    <button class="filter-btn" onclick="filterCards('y1', this)">Year 1</button>
    <button class="filter-btn" onclick="filterCards('y2', this)">Year 2</button>
    <button class="filter-btn" onclick="filterCards('y3', this)">Year 3</button>
    <button class="filter-btn" onclick="filterCards('y4', this)">Year 4</button>
</div>

<div class="legend-row">
    <div class="legend-item">
        <div class="legend-dot" style="background:#c8102e"></div> Core CS Requirement
    </div>
    <div class="legend-item">
        <div class="legend-dot" style="background:#10b981"></div> Required Support Course
    </div>
    <div class="legend-item">
        <div class="legend-dot" style="background:#6366f1"></div> General Education
    </div>
    <div class="legend-item">
        <div class="legend-dot" style="background:#f59e0b"></div> Elective
    </div>
</div>

<div class="progress-section">
    <h2>📊 Degree Progress Overview</h2>

    <div class="progress-row">
        <span class="progress-label">Year 1 — 30 Credits</span>
        <div class="progress-track">
            <div class="progress-fill" style="width:100%; background:#c8102e;"></div>
        </div>
        <span class="progress-val">30 / 30 cr</span>
    </div>

    <div class="progress-row">
        <span class="progress-label">Year 2 — 31 Credits</span>
        <div class="progress-track">
            <div class="progress-fill" style="width:100%; background:#2563eb;"></div>
        </div>
        <span class="progress-val">31 / 31 cr</span>
    </div>

    <div class="progress-row">
        <span class="progress-label">Year 3 — 30 Credits</span>
        <div class="progress-track">
            <div class="progress-fill" style="width:0%; background:#059669;"></div>
        </div>
        <span class="progress-val">0 / 30 cr</span>
    </div>

    <div class="progress-row">
        <span class="progress-label">Year 4 — 31 Credits</span>
        <div class="progress-track">
            <div class="progress-fill" style="width:0%; background:#d97706;"></div>
        </div>
        <span class="progress-val">0 / 31 cr</span>
    </div>
</div>

<div class="semester-grid">

    <div class="semester-card" data-year="y1">
        <div class="card-head">
            <div class="card-head-left">
                <span class="year-badge y1">Year 1</span>
                <h3>Fall Semester</h3>
            </div>
            <span class="credit-badge">15 Credits</span>
        </div>
        <div class="course-list">

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 111</div>
                    <div class="course-name">Introduction to Computer Science I</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-support"></div>
                <div class="course-info">
                    <div class="course-code">MATH 241</div>
                    <div class="course-name">Calculus I</div>
                </div>
                <span class="course-credits">4 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-gen"></div>
                <div class="course-info">
                    <div class="course-code">ENGL 101</div>
                    <div class="course-name">English Composition I</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-gen"></div>
                <div class="course-info">
                    <div class="course-code">HIST 101</div>
                    <div class="course-name">Survey of American History I</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-gen"></div>
                <div class="course-info">
                    <div class="course-code">HLTH 100</div>
                    <div class="course-name">Concepts of Health and Wellness</div>
                </div>
                <span class="course-credits">2 cr</span>
            </div>

        </div>
    </div>

    <div class="semester-card" data-year="y1">
        <div class="card-head">
            <div class="card-head-left">
                <span class="year-badge y1">Year 1</span>
                <h3>Spring Semester</h3>
            </div>
            <span class="credit-badge">15 Credits</span>
        </div>
        <div class="course-list">

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 112</div>
                    <div class="course-name">Introduction to Computer Science II</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-support"></div>
                <div class="course-info">
                    <div class="course-code">MATH 242</div>
                    <div class="course-name">Calculus II</div>
                </div>
                <span class="course-credits">4 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-gen"></div>
                <div class="course-info">
                    <div class="course-code">ENGL 102</div>
                    <div class="course-name">English Composition II</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-gen"></div>
                <div class="course-info">
                    <div class="course-code">HIST 102</div>
                    <div class="course-name">Survey of American History II</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-gen"></div>
                <div class="course-info">
                    <div class="course-code">UNIV 101</div>
                    <div class="course-name">University Experience</div>
                </div>
                <span class="course-credits">2 cr</span>
            </div>

        </div>
    </div>

    <div class="semester-card" data-year="y2">
        <div class="card-head">
            <div class="card-head-left">
                <span class="year-badge y2">Year 2</span>
                <h3>Fall Semester</h3>
            </div>
            <span class="credit-badge">15 Credits</span>
        </div>
        <div class="course-list">

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 211</div>
                    <div class="course-name">Data Structures</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 251</div>
                    <div class="course-name">Computer Organization and Assembly Language</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-support"></div>
                <div class="course-info">
                    <div class="course-code">MATH 311</div>
                    <div class="course-name">Discrete Mathematics</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-support"></div>
                <div class="course-info">
                    <div class="course-code">MATH 243</div>
                    <div class="course-name">Calculus III</div>
                </div>
                <span class="course-credits">4 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-gen"></div>
                <div class="course-info">
                    <div class="course-code">COMM 103</div>
                    <div class="course-name">Introduction to Public Speaking</div>
                </div>
                <span class="course-credits">2 cr</span>
            </div>

        </div>
    </div>

    <div class="semester-card" data-year="y2">
        <div class="card-head">
            <div class="card-head-left">
                <span class="year-badge y2">Year 2</span>
                <h3>Spring Semester</h3>
            </div>
            <span class="credit-badge">16 Credits</span>
        </div>
        <div class="course-list">

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 311</div>
                    <div class="course-name">Design and Analysis of Algorithms</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 341</div>
                    <div class="course-name">Object-Oriented Programming</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-support"></div>
                <div class="course-info">
                    <div class="course-code">MATH 321</div>
                    <div class="course-name">Linear Algebra</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-support"></div>
                <div class="course-info">
                    <div class="course-code">PHYS 201</div>
                    <div class="course-name">General Physics I with Lab</div>
                </div>
                <span class="course-credits">4 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-gen"></div>
                <div class="course-info">
                    <div class="course-code">AFST 201</div>
                    <div class="course-name">Introduction to African American Studies</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

        </div>
    </div>

    <div class="semester-card" data-year="y3">
        <div class="card-head">
            <div class="card-head-left">
                <span class="year-badge y3">Year 3</span>
                <h3>Fall Semester</h3>
            </div>
            <span class="credit-badge">15 Credits</span>
        </div>
        <div class="course-list">

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 411</div>
                    <div class="course-name">Operating Systems</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 421</div>
                    <div class="course-name">Database Management Systems</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 431</div>
                    <div class="course-name">Software Engineering I</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-elective"></div>
                <div class="course-info">
                    <div class="course-code">COSC 4XX</div>
                    <div class="course-name">CS Elective (e.g. Machine Learning)</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-gen"></div>
                <div class="course-info">
                    <div class="course-code">PHIL 315</div>
                    <div class="course-name">Ethics in Technology</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

        </div>
    </div>

    <div class="semester-card" data-year="y3">
        <div class="card-head">
            <div class="card-head-left">
                <span class="year-badge y3">Year 3</span>
                <h3>Spring Semester</h3>
            </div>
            <span class="credit-badge">15 Credits</span>
        </div>
        <div class="course-list">

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 412</div>
                    <div class="course-name">Computer Networks</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 432</div>
                    <div class="course-name">Software Engineering II</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-elective"></div>
                <div class="course-info">
                    <div class="course-code">COSC 4XX</div>
                    <div class="course-name">CS Elective (e.g. Cybersecurity)</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-support"></div>
                <div class="course-info">
                    <div class="course-code">STAT 351</div>
                    <div class="course-name">Probability and Statistics</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-gen"></div>
                <div class="course-info">
                    <div class="course-code">GEN ED</div>
                    <div class="course-name">Humanities / Fine Arts Elective</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

        </div>
    </div>

    <div class="semester-card" data-year="y4">
        <div class="card-head">
            <div class="card-head-left">
                <span class="year-badge y4">Year 4</span>
                <h3>Fall Semester</h3>
            </div>
            <span class="credit-badge">15 Credits</span>
        </div>
        <div class="course-list">

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 490</div>
                    <div class="course-name">Senior Project I</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 461</div>
                    <div class="course-name">Programming Languages</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 471</div>
                    <div class="course-name">Theory of Computation</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-elective"></div>
                <div class="course-info">
                    <div class="course-code">COSC 4XX</div>
                    <div class="course-name">CS Elective (e.g. Computer Vision)</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-gen"></div>
                <div class="course-info">
                    <div class="course-code">GEN ED</div>
                    <div class="course-name">Social Sciences Elective</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

        </div>
    </div>

    <div class="semester-card" data-year="y4">
        <div class="card-head">
            <div class="card-head-left">
                <span class="year-badge y4">Year 4</span>
                <h3>Spring Semester</h3>
            </div>
            <span class="credit-badge">16 Credits</span>
        </div>
        <div class="course-list">

            <div class="course-item">
                <div class="course-dot dot-core"></div>
                <div class="course-info">
                    <div class="course-code">COSC 491</div>
                    <div class="course-name">Senior Project II (Capstone)</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-elective"></div>
                <div class="course-info">
                    <div class="course-code">COSC 4XX</div>
                    <div class="course-name">CS Elective (Free Choice)</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-elective"></div>
                <div class="course-info">
                    <div class="course-code">COSC 4XX</div>
                    <div class="course-name">CS Elective (Free Choice)</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-gen"></div>
                <div class="course-info">
                    <div class="course-code">GEN ED</div>
                    <div class="course-name">Upper-Level Writing Requirement</div>
                </div>
                <span class="course-credits">3 cr</span>
            </div>

            <div class="course-item">
                <div class="course-dot dot-gen"></div>
                <div class="course-info">
                    <div class="course-code">GEN ED</div>
                    <div class="course-name">Free Elective</div>
                </div>
                <span class="course-credits">4 cr</span>
            </div>

        </div>
    </div>


</div><!-- end .semester-grid -->


<script>
function filterCards(year, btn){
    document.querySelectorAll(".filter-btn").forEach(function(b){
        b.classList.remove("active")
    })
    btn.classList.add("active")
    document.querySelectorAll(".semester-card").forEach(function(card){
        if(year === "all" || card.dataset.year === year){
            card.style.display = ""
        } else {
            card.style.display = "none"
        }
    })
}
</script>

</body>
</html>