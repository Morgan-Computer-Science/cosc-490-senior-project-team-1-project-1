<?php

require_once("db.php");

$success = false;

if(isset($_POST['email'])){

$email = $_POST['email'];
$password = password_hash($_POST['password'], PASSWORD_DEFAULT);

$stmt = $conn->prepare("UPDATE students SET password=? WHERE email=?");
$stmt->bind_param("ss",$password,$email);
$stmt->execute();

$success = true;

}

?>

<!DOCTYPE html>
<html>

<head>

<title>Reset Password</title>
<link rel="stylesheet" href="styles.css">

</head>

<body>

<div class="stage">

<div class="laptop">

<div class="laptop-top"></div>

<div class="screen-shell">

<div class="screen-scale">

<div class="screen-ui">

<div class="auth-panel">

<div class="auth-inner">

<div class="brand">
<img src="assets/logo.png" style="height:40px">
</div>

<div class="form-wrap">

<h1>Reset your password</h1>

<p class="subtitle">
Enter your email and create a new password
</p>

<div class="panel-divider"></div>

<form class="login-form" method="POST">

<label class="field-label">Email</label>
<input class="input" type="email" name="email" required>

<label class="field-label">Create New Password</label>
<input class="input" type="password" name="password" required>

<label class="field-label">Re-type Password</label>
<input class="input" type="password" name="confirm_password" required>

<button class="btn-login">Reset Password</button>

</form>

<?php if($success): ?>

<div class="success-popup">
<h2>Thank you!</h2>
<p>Your Password has been updated, please login with your new credentials</p>
</div>

<?php endif; ?>

<p class="signup-text">
<a href="index.php">Back to Login</a>
</p>

</div>

</div>

</div>

<div class="hero-panel">

<div class="hero-content">

<h2>
<span>Password Reset</span>
<span>Secure your account.</span>
</h2>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="laptop-base"></div>

</div>

</div>

</body>

</html>