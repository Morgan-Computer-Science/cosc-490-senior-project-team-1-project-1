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
    CHANGE FROM ORIGINAL:
    Wrapped the logo in an <a> tag so clicking it goes to curriculum.php
-->
<a href="curriculum.php">
<img src="assets/logo.png" class="logo">
</a>

<div class="chat-header-icons">

<img src="assets/help_button.png" onclick="openSupport()" class="icon">

<a href="settings.php">
<img src="assets/user_icon.png" class="icon">
</a>

</div>

</div>


<!-- CHAT TITLE -->

<div class="chat-center">

<h1 class="chat-title">Good Afternoon <?php echo htmlspecialchars($_SESSION["name"]); ?>!</h1>

<p class="chat-subtitle">How can I help you today?</p>

<div id="chatBox" class="chat-area"></div>

</div>


<!-- ATTACHMENT PREVIEW -->

<div id="attachmentPreview" class="attachment-preview"></div>


<!-- CHAT INPUT -->

<div class="chat-input">

<img src="assets/add_file_button.png" onclick="toggleMenu()" class="plus">

<input id="userInput" placeholder="Ask anything" onkeydown="handleEnter(event)">

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

/* ── MENU ─────────────────────────────────────── */

function toggleMenu(){
    document.getElementById("plusMenu").classList.toggle("show")
}


/* ── SUPPORT POPUP ────────────────────────────── */

function openSupport(){
    document.getElementById("supportPopup").style.display = "block"
}

function closeSupport(){
    document.getElementById("supportPopup").style.display = "none"
}


/* ── EMAIL ADVISOR POPUP ──────────────────────── */

function emailAdvisor(){
    document.getElementById("emailPopup").style.display = "block"
}

function closeAdvisor(){
    document.getElementById("emailPopup").style.display = "none"
}


/* ── ATTACHMENT STORAGE ───────────────────────── */

let attachedFile = null


/* ── ENTER KEY SUPPORT ────────────────────────── */

/*
    CHANGE FROM ORIGINAL:
    Added this function so pressing Enter sends the message,
    just like clicking the send button.
*/
function handleEnter(event){
    if(event.key === "Enter"){
        event.preventDefault()
        sendMessage()
    }
}


/* ── SEND MESSAGE ─────────────────────────────── */

/*
    CHANGES FROM ORIGINAL:
    1. Added a loading/typing indicator so the UI doesn't
       look frozen while waiting for the AI response.
    2. Disabled the send button while the request is in
       flight so the student can't double-send.
    3. Proper error handling — if the request fails, shows
       a friendly message instead of silently breaking.
    4. The rest of the logic (FormData, fetch, response
       parsing) is unchanged — it already matched chat.php.
*/
async function sendMessage(){

    let input   = document.getElementById("userInput")
    let message = input.value.trim()

    if(message === "" && !attachedFile) return

    let chat    = document.getElementById("chatBox")
    let sendBtn = document.getElementById("sendBtn")

    /* show user message */
    if(message !== ""){
        chat.innerHTML += `<div class="user-msg">${escapeHtml(message)}</div>`
    }

    /* show attachment label in chat */
    if(attachedFile){
        chat.innerHTML += `<div class="user-msg">📎 ${escapeHtml(attachedFile.name)}</div>`
    }

    input.value = ""

    /* scroll to bottom */
    chat.scrollTop = chat.scrollHeight

    /* disable send button while waiting */
    sendBtn.style.opacity = "0.5"
    sendBtn.style.pointerEvents = "none"
    input.disabled = true

    /* show typing indicator */
    let loadingId = "loading-" + Date.now()
    chat.innerHTML += `
        <div class="ai-msg" id="${loadingId}">
            <span class="dot-one">.</span>
            <span class="dot-two">.</span>
            <span class="dot-three">.</span>
        </div>`
    chat.scrollTop = chat.scrollHeight

    /* build form data — chat.php reads $_POST["message"] and $_FILES["file"] */
    let form = new FormData()
    form.append("message", message)
    if(attachedFile){
        form.append("file", attachedFile)
    }

    try {

        let res  = await fetch("chat.php", { method: "POST", body: form })
        let data = await res.json()

        /* remove typing indicator */
        let loadingEl = document.getElementById(loadingId)
        if(loadingEl) loadingEl.remove()

        /* extract reply from Gemini response format */
        let reply = data?.candidates?.[0]?.content?.parts?.[0]?.text

        if(!reply){
            reply = "I'm having trouble responding right now. Please try again. ⏳"
        }

        /* format reply for display */
        reply = reply
            .replace(/\n\n/g, "<br><br>")
            .replace(/\n/g,   "<br>")
            .replace(/\*\*/g, "")

        chat.innerHTML += `<div class="ai-msg">${reply}</div>`

    } catch(err) {

        /* remove typing indicator */
        let loadingEl = document.getElementById(loadingId)
        if(loadingEl) loadingEl.remove()

        chat.innerHTML += `<div class="ai-msg">⚠️ Could not reach Morgan AI. Please check your connection and try again.</div>`

        console.error("Morgan AI fetch error:", err)

    } finally {

        /* always re-enable input after response */
        sendBtn.style.opacity = ""
        sendBtn.style.pointerEvents = ""
        input.disabled = false
        input.focus()

        chat.scrollTop = chat.scrollHeight
        saveChat()

    }

    /* clear attachment */
    attachedFile = null
    document.getElementById("attachmentPreview").innerHTML = ""
}


/* ── XSS PROTECTION ───────────────────────────── */

/*
    CHANGE FROM ORIGINAL:
    Escaping user input before putting it in innerHTML
    prevents malicious scripts from running in the chat.
*/
function escapeHtml(str){
    return str
        .replace(/&/g,  "&amp;")
        .replace(/</g,  "&lt;")
        .replace(/>/g,  "&gt;")
        .replace(/"/g,  "&quot;")
        .replace(/'/g,  "&#039;")
}


/* ── FILE UPLOAD ──────────────────────────────── */

function uploadFile(){

    let input = document.createElement("input")
    input.type = "file"

    input.onchange = function(){
        attachedFile = input.files[0]
        let preview  = document.getElementById("attachmentPreview")
        preview.innerHTML = `<div class="attachment-item">📎 ${escapeHtml(attachedFile.name)}</div>`
    }

    input.click()
    document.getElementById("plusMenu").classList.remove("show")
}


/* ── IMAGE UPLOAD ─────────────────────────────── */

function uploadImage(){

    let input    = document.createElement("input")
    input.type   = "file"
    input.accept = "image/*"

    input.onchange = function(){
        attachedFile = input.files[0]
        let preview  = document.getElementById("attachmentPreview")
        preview.innerHTML = `<div class="attachment-item">🖼 ${escapeHtml(attachedFile.name)}</div>`
    }

    input.click()
    document.getElementById("plusMenu").classList.remove("show")
}


/* ── SAVE / LOAD CHAT ─────────────────────────── */

function saveChat(){
    let chat = document.getElementById("chatBox").innerHTML
    sessionStorage.setItem("chatHistory", chat)
}

function loadChat(){
    let saved = sessionStorage.getItem("chatHistory")
    if(saved){
        document.getElementById("chatBox").innerHTML = saved
        let chat = document.getElementById("chatBox")
        chat.scrollTop = chat.scrollHeight
    }
}

window.onload = loadChat


/* ── CONTACT SUPPORT ──────────────────────────── */

async function submitSupport(){

    let name    = document.getElementById("supportName").value
    let email   = document.getElementById("supportEmail").value
    let message = document.getElementById("supportMessage").value

    let form = new FormData()
    form.append("name",    name)
    form.append("email",   email)
    form.append("message", message)

    await fetch("contact_support.php", { method: "POST", body: form })

    alert("Support request sent")
    closeSupport()
}


/* ── EMAIL ADVISOR ────────────────────────────── */

async function sendAdvisorEmail(){

    let name    = document.getElementById("advisorName").value
    let email   = document.getElementById("advisorEmail").value
    let message = document.getElementById("advisorMessage").value

    let form = new FormData()
    form.append("name",    name)
    form.append("email",   email)
    form.append("message", message)

    await fetch("email_advisor.php", { method: "POST", body: form })

    alert("Message sent to advisor")
    closeAdvisor()
}

</script>

<!--
    CHANGE FROM ORIGINAL:
    Added CSS for the typing indicator dots animation.
    Everything else matches your existing styles.css.
-->
<style>
.dot-one, .dot-two, .dot-three {
    display: inline-block;
    font-size: 1.6rem;
    line-height: 1;
    animation: dotBounce .9s infinite ease-in-out;
    color: #aaa;
}
.dot-two { animation-delay: .15s; }
.dot-three { animation-delay: .30s; }

@keyframes dotBounce {
    0%, 80%, 100% { transform: translateY(0);    opacity: .4; }
    40%           { transform: translateY(-6px);  opacity: 1;  }
}
</style>

</body>
</html>