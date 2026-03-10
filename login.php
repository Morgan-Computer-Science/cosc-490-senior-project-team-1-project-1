<?php

session_start();
require_once("db.php");

$email = $_POST["email"];
$password = $_POST["password"];

/* Query student table */

$sql = "SELECT * FROM students WHERE email=?";
$stmt = $conn->prepare($sql);
$stmt->bind_param("s",$email);
$stmt->execute();

$result = $stmt->get_result();

if($result->num_rows == 1){

    $user = $result->fetch_assoc();

    if(password_verify($password,$user["password"])){

        /* Save session variables */

        $_SESSION["user_id"] = $user["id"];
        $_SESSION["name"] = $user["first_name"];
        $_SESSION["student_id"] = $user["student_id"];
        $_SESSION["email"] = $user["email"];

        header("Location: dashboard.php");
        exit();

    } else {

        echo "Incorrect password";

    }

} else {

    echo "User not found";

}

?>