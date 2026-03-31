<?php
session_start();

/* Redirect if not logged in */
if(!isset($_SESSION["user_id"])) {
    header("Location: index.php");
    exit();
}

/* Get user data */
$name = $_SESSION["name"];
$email = $_SESSION["email"];
$student_id = $_SESSION["student_id"];
?>

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Main Dashboard</title>

<style>
body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f6f9;
}

/* Navbar */
.navbar {
    background: #1e3a8a;
    color: white;
    padding: 15px 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Layout */
.wrapper {
    display: flex;
}

/* Sidebar */
.sidebar {
    width: 220px;
    background: #111827;
    color: white;
    height: 100vh;
    padding-top: 20px;
}

.sidebar a {
    display: block;
    color: white;
    padding: 12px 20px;
    text-decoration: none;
}

.sidebar a:hover {
    background: #1f2937;
}

/* Main content */
.content {
    flex: 1;
    padding: 30px;
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Grid */
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
}

/* Logout button */
.logout-btn {
    background: #ef4444;
    color: white;
    border: none;
    padding: 8px 15px;
    cursor: pointer;
    border-radius: 5px;
}
</style>
</head>

<body>

<!-- NAVBAR -->
<div class="navbar">
    <h2>Welcome, <?php echo htmlspecialchars($name); ?> 👋</h2>

    <form action="logout.php" method="POST">
        <button class="logout-btn">Logout</button>
    </form>
</div>

<div class="wrapper">

    <!-- SIDEBAR -->
    <div class="sidebar">
        <a href="#">Dashboard</a>
        <a href="#">Profile</a>
        <a href="#">Courses</a>
        <a href="#">Settings</a>
    </div>

    <!-- MAIN CONTENT -->
    <div class="content">

        <div class="card">
            <h2>Your Info</h2>
            <p><strong>Email:</strong> <?php echo htmlspecialchars($email); ?></p>
            <p><strong>Student ID:</strong> <?php echo htmlspecialchars($student_id); ?></p>
        </div>

        <div class="grid">

            <div class="card">
                <h3>Courses</h3>
                <p>View your enrolled courses.</p>
            </div>

            <div class="card">
                <h3>Progress</h3>
                <p>Track your academic progress.</p>
            </div>

            <div class="card">
                <h3>Notifications</h3>
                <p>Stay updated with alerts.</p>
            </div>

            <div class="card">
                <h3>Settings</h3>
                <p>Manage your account.</p>
            </div>

        </div>

    </div>

</div>

</body>
</html>