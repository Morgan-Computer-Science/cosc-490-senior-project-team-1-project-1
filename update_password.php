<?php

require_once("db.php");

$email = $_POST['email'];
$password = $_POST['password'];
$confirm = $_POST['confirm_password'];

if($password !== $confirm){
die("Passwords do not match");
}

$hash = password_hash($password, PASSWORD_DEFAULT);

$stmt = $conn->prepare("UPDATE users SET password=? WHERE email=?");
$stmt->bind_param("ss",$hash,$email);
$stmt->execute();

header("Location: index.php");
exit;

?>