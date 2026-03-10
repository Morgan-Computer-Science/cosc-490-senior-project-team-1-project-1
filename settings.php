<?php
session_start();
require_once("db.php");

/* Prevent access if not logged in */
if(!isset($_SESSION["user_id"])){
    header("Location: index.php");
    exit();
}

$id = $_SESSION["user_id"];

/* Use students table instead of users */
$stmt = $conn->prepare("SELECT first_name,last_name,email,student_id FROM students WHERE id=?");
$stmt->bind_param("i",$id);
$stmt->execute();

$result = $stmt->get_result();
$user = $result->fetch_assoc();
?>

<!DOCTYPE html>
<html>

<head>

<title>Settings</title>

<link rel="stylesheet" href="styles.css">

<style>

.settings-container{
width:1280px;
height:832px;
margin:auto;
position:relative;
background:#1E1E1E;
color:white;
font-family:'Inknut Antiqua',serif;
border:1px solid #2a2a2a;
}

.settings-title{
font-size:40px;
position:absolute;
left:80px;
top:150px;
}

.settings-profile{
position:absolute;
left:80px;
top:250px;
display:flex;
align-items:center;
gap:20px;
}

.settings-profile img{
width:80px;
border-radius:50%;
}

.student-name{
font-size:30px;
}

.student-info{
font-size:20px;
opacity:.85;
}

.edit-profile{
position:absolute;
left:693px;
top:304px;
width:491px;
height:54px;
background:#D9D9D9;
border:none;
color:black;
font-size:18px;
border-radius:30px;
cursor:pointer;
}

.settings-columns{
position:absolute;
top:420px;
left:80px;
display:flex;
gap:400px;
font-size:20px;
}

.settings-section-title{
font-weight:bold;
margin-bottom:15px;
border-bottom:1px solid #3a3a3a;
padding-bottom:5px;
}

.settings-item{
margin-bottom:15px;
cursor:pointer;
}

.settings-item:hover{
opacity:.8;
}

.logout-btn{
position:absolute;
left:1077px;
top:769px;
width:153px;
height:44px;
background:#CA5959;
color:#1E1E1E;
border:none;
border-radius:12px;
font-size:18px;
cursor:pointer;
}

</style>

</head>

<body>

<div class="settings-container">

<img src="assets/logo.png" class="logo">

<a href="dashboard.php" style="position:absolute;right:60px;top:20px;color:white">
&lt; Back to Chat
</a>

<h1 class="settings-title">Settings</h1>

<div class="settings-profile">

<img src="assets/user_avatar.png">

<div>

<div class="student-name">
<?php echo $user["first_name"] . " " . $user["last_name"]; ?>
</div>

<div class="student-info">
<?php echo $user["email"]; ?>
</div>

<div class="student-info">
ID: <?php echo $user["student_id"]; ?>
</div>

</div>

</div>

<button class="edit-profile">Edit Profile</button>

<div class="settings-columns">

<div>

<div class="settings-section-title">Account</div>

<div class="settings-item">Change Password ></div>
<div class="settings-item">Update Email ></div>
<div class="settings-item">Preference ></div>
<div class="settings-item">Notifications ></div>

</div>

<div>

<div class="settings-section-title">Academic Settings</div>

<div class="settings-item">Graduation ></div>
<div class="settings-item">Assigned Advisor ></div>
<div class="settings-item">Degree Track ></div>

</div>

</div>

<button class="logout-btn" onclick="logout()">Logout</button>

</div>

<script>

function logout(){
window.location="logout.php"
}

</script>

</body>
</html>