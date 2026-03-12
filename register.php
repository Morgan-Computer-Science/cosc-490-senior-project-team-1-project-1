<!DOCTYPE html>
<html>

<head>

<title>Create Account</title>

<link rel="stylesheet" href="styles.css">

</head>

<body>

<div class="stage">

<div class="laptop">

<div class="screen-shell">

<div class="screen-ui">

<!-- LEFT SIDE -->

<div class="auth-panel">

<div class="form-wrap">

<img src="assets/logo.png" style="width:160px;margin-bottom:20px;">

<h1>Create your account</h1>

<form class="login-form" method="POST" action="register_user.php" onsubmit="return validatePasswords()">

<label class="field-label">First Name</label>
<input class="input" name="first_name" required>

<label class="field-label">Last Name</label>
<input class="input" name="last_name" required>

<label class="field-label">Student ID #</label>
<input class="input" name="student_id" required>

<label class="field-label">School Year</label>

<select class="input" name="school_year" required>

<option value="">Select School Year</option>
<option value="Frehman">Freshman</option>

<option value="Sophomore">Sophomore</option>

<option value="Junior">Junior</option>

<option value="Senior">Senior</option>

</select>

<label class="field-label">Date of Birth</label>
<input class="input" type="date" name="dob" required>

<label class="field-label">School Email</label>
<input class="input" type="email" name="email" required>

<label class="field-label">Create Password</label>
<input class="input" type="password" id="password" name="password" required>

<label class="field-label">Re-type Password</label>
<input class="input" type="password" id="confirm_password" name="confirm_password" required>

<button class="btn-login">Sign Up</button>

</form>

<a href="index.php" class="signup-text">Back to Login</a>

</div>

</div>

<!-- RIGHT SIDE -->

<div class="hero-panel">

<img src="assets/banner.png" class="hero-banner">

</div>

</div>

</div>

</div>

</div>

<script>

function validatePasswords(){

let pass=document.getElementById("password").value
let confirm=document.getElementById("confirm_password").value

if(pass!==confirm){

alert("Passwords do not match")
return false

}

return true

}

</script>

</body>

</html>