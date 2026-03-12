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

<script>
async function settingsApi(payload){
const formData = new FormData()
Object.keys(payload).forEach((key)=>formData.append(key,payload[key]))

const res = await fetch("settings_actions.php",{
method:"POST",
body:formData
})

const data = await res.json()
if(!data.ok){
throw new Error(data.message || "Action failed")
}
return data
}

function normalizeItemText(text){
return text.replace(">", "").trim().toLowerCase()
}

async function onEditProfile(){
try{
const data = await settingsApi({action:"load_settings"})
const current = data.profile || {}

const first = prompt("First name", current.first_name || "")
if(first === null){ return }
const last = prompt("Last name", current.last_name || "")
if(last === null){ return }
const studentId = prompt("Student ID", current.student_id || "")
if(studentId === null){ return }

await settingsApi({
action:"edit_profile",
first_name:first,
last_name:last,
student_id:studentId
})
alert("Profile updated.")
window.location.reload()
}
catch(err){
alert(err.message)
}
}

async function onChangePassword(){
try{
const currentPassword = prompt("Enter current password")
if(currentPassword === null){ return }
const newPassword = prompt("Enter new password (at least 8 characters)")
if(newPassword === null){ return }
const confirmPassword = prompt("Confirm new password")
if(confirmPassword === null){ return }

await settingsApi({
action:"change_password",
current_password:currentPassword,
new_password:newPassword,
confirm_password:confirmPassword
})
alert("Password updated.")
}
catch(err){
alert(err.message)
}
}

async function onUpdateEmail(){
try{
const data = await settingsApi({action:"load_settings"})
const email = prompt("Enter new email (@morgan.edu)", (data.profile && data.profile.email) ? data.profile.email : "")
if(email === null){ return }

await settingsApi({
action:"update_email",
email:email
})
alert("Email updated.")
window.location.reload()
}
catch(err){
alert(err.message)
}
}

async function onPreference(){
try{
const data = await settingsApi({action:"load_settings"})
const theme = prompt("Preference theme: dark, light, or system", (data.settings && data.settings.theme) ? data.settings.theme : "dark")
if(theme === null){ return }

await settingsApi({
action:"preference",
theme:theme.toLowerCase()
})
alert("Preference updated.")
}
catch(err){
alert(err.message)
}
}

async function onNotifications(){
try{
const data = await settingsApi({action:"load_settings"})
const currentlyEnabled = data.settings && parseInt(data.settings.notifications,10) === 1
const enable = confirm((currentlyEnabled ? "Notifications are ON." : "Notifications are OFF.") + " Click OK to set ON, Cancel to set OFF.")

await settingsApi({
action:"notifications",
enabled:enable ? "1" : "0"
})
alert("Notification setting updated.")
}
catch(err){
alert(err.message)
}
}

async function onGraduation(){
try{
const data = await settingsApi({action:"load_settings"})
const graduation = prompt("Enter graduation term/year (example: Spring 2027)", (data.academic && data.academic.graduation) ? data.academic.graduation : "")
if(graduation === null){ return }

await settingsApi({
action:"graduation",
graduation:graduation
})
alert("Graduation updated.")
}
catch(err){
alert(err.message)
}
}

async function onAssignedAdvisor(){
try{
const data = await settingsApi({action:"load_settings"})
const advisor = prompt("Enter assigned advisor name", (data.academic && data.academic.assigned_advisor) ? data.academic.assigned_advisor : "")
if(advisor === null){ return }

await settingsApi({
action:"assigned_advisor",
advisor:advisor
})
alert("Assigned advisor updated.")
}
catch(err){
alert(err.message)
}
}

async function onDegreeTrack(){
try{
const data = await settingsApi({action:"load_settings"})
const track = prompt("Enter degree track", (data.academic && data.academic.degree_track) ? data.academic.degree_track : "")
if(track === null){ return }

await settingsApi({
action:"degree_track",
degree_track:track
})
alert("Degree track updated.")
}
catch(err){
alert(err.message)
}
}

document.addEventListener("DOMContentLoaded", function(){
const editBtn = document.querySelector(".edit-profile")
if(editBtn){
editBtn.addEventListener("click", onEditProfile)
}

document.querySelectorAll(".settings-item").forEach((item)=>{
item.addEventListener("click", function(){
const label = normalizeItemText(item.textContent || "")

if(label === "change password"){ onChangePassword(); return }
if(label === "update email"){ onUpdateEmail(); return }
if(label === "preference"){ onPreference(); return }
if(label === "notifications"){ onNotifications(); return }
if(label === "graduation"){ onGraduation(); return }
if(label === "assigned advisor"){ onAssignedAdvisor(); return }
if(label === "degree track"){ onDegreeTrack(); return }
})
})
})
</script>

</body>
</html>
