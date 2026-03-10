<?php

session_start();
require_once "db.php";

$name=$_POST['name'];
$email=$_POST['email'];
$message=$_POST['message'];

$sql="INSERT INTO support_messages(name,email,message)
VALUES(?,?,?)";

$stmt=$conn->prepare($sql);

$stmt->bind_param("sss",$name,$email,$message);

$stmt->execute();

echo "Message sent";

?>