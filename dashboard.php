<?php
session_start();

if(!isset($_SESSION["user_id"])){
    header("Location: index.php");
    exit();
}
?>

<!DOCTYPE html>
<html>

<head>
<title>Morgan AI Chat</title>
<link rel="stylesheet" href="styles.css">
</head>

<body>

<div class="chat-page">

<!-- HEADER -->

<div class="chat-header">

<!--
    CHANGE 1: Wrapped logo in <a> tag so clicking it goes to curriculum.php
    Before: <img src="assets/logo.png" class="logo">
    After:  <a href="curriculum.php"><img src="assets/logo.png" class="logo"></a>
-->
<a href="curriculum.php"><img src="assets/logo.png" class="logo"></a>

<div class="chat-header-icons">

<img src="assets/help_button.png" onclick="openSupport()" class="icon">

<a href="settings.php">
<img src="assets/user_icon.png" class="icon">
</a>

</div>

</div>


<!-- CHAT TITLE -->

<div class="chat-center">

<!--
    CHANGE 2: Added htmlspecialchars() to prevent XSS
    Before: <?php echo $_SESSION["name"]; ?>
    After:  <?php echo htmlspecialchars($_SESSION["name"]); ?>
-->
<h1 class="chat-title">Good Afternoon <?php echo htmlspecialchars($_SESSION["name"]); ?>!</h1>

<p class="chat-subtitle">How can I help you today?</p>

<div id="chatBox" class="chat-area"></div>

</div>


<!-- ATTACHMENT PREVIEW -->

<div id="attachmentPreview" class="attachment-preview"></div>


<!-- CHAT INPUT -->

<div class="chat-input">

<img src="assets/add_file_button.png" onclick="toggleMenu()" class="plus">

<input id="userInput" placeholder="Ask anything" onkeydown="if(event.key==='Enter') sendMessage()">

<img src="assets/send_button.png" onclick="sendMessage()" class="send" id="sendBtn">

</div>


<!-- PLUS MENU -->

<div id="plusMenu" class="plus-menu">

<div class="menu-item" onclick="uploadFile()">
<img src="assets/file_icon.png" class="menu-icon">
Upload File
</div>

<div class="menu-item" onclick="uploadImage()">
<img src="assets/image_icon.png" class="menu-icon">
Upload Image
</div>

<div class="menu-item" onclick="emailAdvisor()">
<img src="assets/mail_icon.png" class="menu-icon">
Email Advisor
</div>

</div>


<!-- CONTACT SUPPORT POPUP -->

<div id="supportPopup" class="popup">

<h2>Contact Us</h2>

<input id="supportName" placeholder="First & Last Name">

<input id="supportEmail" placeholder="Email">

<textarea id="supportMessage" placeholder="Message"></textarea>

<button onclick="submitSupport()">Send</button>

<button onclick="closeSupport()">Close</button>

</div>


<!-- EMAIL ADVISOR POPUP -->

<div id="emailPopup" class="popup">

<h2>Email Advisor</h2>

<input id="advisorName" placeholder="Your Name">

<input id="advisorEmail" placeholder="Your Email">

<textarea id="advisorMessage" placeholder="Message"></textarea>

<button onclick="sendAdvisorEmail()">Send</button>

<button onclick="closeAdvisor()">Close</button>

</div>


<script>

/* MENU */

function toggleMenu(){
document.getElementById("plusMenu").classList.toggle("show")
}


/* SUPPORT POPUP */

function openSupport(){
document.getElementById("supportPopup").style.display="block"
}

function closeSupport(){
document.getElementById("supportPopup").style.display="none"
}


/* EMAIL ADVISOR */

function emailAdvisor(){
document.getElementById("emailPopup").style.display="block"
}

function closeAdvisor(){
document.getElementById("emailPopup").style.display="none"
}


/* ATTACHMENT STORAGE */

let attachedFile = null;


/* SEND MESSAGE */

async function sendMessage(){

let input   = document.getElementById("userInput")
let message = input.value.trim()
let sendBtn = document.getElementById("sendBtn")

if(message === "" && !attachedFile) return

let chat = document.getElementById("chatBox")

/* show user message */
chat.innerHTML += `<div class="user-msg">${message}</div>`

/* show attachment in chat */
if(attachedFile){
chat.innerHTML += `<div class="user-msg">📎 ${attachedFile.name}</div>`
}

input.value = ""

/* disable send while waiting */
sendBtn.style.opacity = "0.5"
sendBtn.style.pointerEvents = "none"
input.disabled = true

/* show typing indicator */
let loadingId = "loading-" + Date.now()
chat.innerHTML += `<div class="ai-msg" id="${loadingId}">
    <span style="animation:dot .9s infinite ease-in-out;display:inline-block">.</span>
    <span style="animation:dot .9s .15s infinite ease-in-out;display:inline-block">.</span>
    <span style="animation:dot .9s .3s infinite ease-in-out;display:inline-block">.</span>
</div>`
chat.scrollTop = chat.scrollHeight

let form = new FormData()
form.append("message", message)
if(attachedFile){
form.append("file", attachedFile)
}

try {

/* send to chat.php */
let res  = await fetch("chat.php", { method:"POST", body:form })
let data = await res.json()

/* remove typing indicator */
document.getElementById(loadingId)?.remove()

let reply = data?.candidates?.[0]?.content?.parts?.[0]?.text || "Morgan AI is unavailable right now. Please try again."

reply = reply
.replace(/\n\n/g,"<br><br>")
.replace(/\n/g,"<br>")
.replace(/\*\*/g,"")

chat.innerHTML += `<div class="ai-msg">${reply}</div>`

} catch(err) {

document.getElementById(loadingId)?.remove()
chat.innerHTML += `<div class="ai-msg">⚠️ Could not reach Morgan AI. Please check your connection.</div>`

}

/* re-enable input */
sendBtn.style.opacity = ""
sendBtn.style.pointerEvents = ""
input.disabled = false
input.focus()

chat.scrollTop = chat.scrollHeight

/* clear attachment */
attachedFile = null
document.getElementById("attachmentPreview").innerHTML = ""

saveChat()

}


/* FILE UPLOAD */

function uploadFile(){

let input=document.createElement("input")
input.type="file"

input.onchange=function(){
attachedFile=input.files[0]
document.getElementById("attachmentPreview").innerHTML=`<div class="attachment-item">📎 ${attachedFile.name}</div>`
}

input.click()
document.getElementById("plusMenu").classList.remove("show")
}


/* IMAGE UPLOAD */

function uploadImage(){

let input=document.createElement("input")
input.type="file"
input.accept="image/*"

input.onchange=function(){
attachedFile=input.files[0]
document.getElementById("attachmentPreview").innerHTML=`<div class="attachment-item">🖼 ${attachedFile.name}</div>`
}

input.click()
document.getElementById("plusMenu").classList.remove("show")
}


/* SAVE CHAT */

function saveChat(){
let chat=document.getElementById("chatBox").innerHTML
sessionStorage.setItem("chatHistory",chat)
}


/* LOAD CHAT */

function loadChat(){
let saved=sessionStorage.getItem("chatHistory")
if(saved){
document.getElementById("chatBox").innerHTML=saved
let chat=document.getElementById("chatBox")
chat.scrollTop=chat.scrollHeight
}
}

window.onload=loadChat


/* CONTACT SUPPORT */

async function submitSupport(){

let name=document.getElementById("supportName").value
let email=document.getElementById("supportEmail").value
let message=document.getElementById("supportMessage").value

let form=new FormData()
form.append("name",name)
form.append("email",email)
form.append("message",message)

await fetch("contact_support.php",{method:"POST",body:form})

alert("Support request sent")
closeSupport()
}


/* EMAIL ADVISOR */

async function sendAdvisorEmail(){

let name=document.getElementById("advisorName").value
let email=document.getElementById("advisorEmail").value
let message=document.getElementById("advisorMessage").value

let form=new FormData()
form.append("name",name)
form.append("email",email)
form.append("message",message)

await fetch("email_advisor.php",{method:"POST",body:form})

alert("Message sent to advisor")
closeAdvisor()
}

</script>

<!-- Typing indicator animation -->
<style>
@keyframes dot {
  0%, 80%, 100% { transform: translateY(0);   opacity: .4; }
  40%           { transform: translateY(-5px); opacity: 1;  }
}
</style>

</body>
</html>