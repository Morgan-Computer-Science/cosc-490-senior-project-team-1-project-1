<?php
session_start();
require_once("db.php");

// Only run login logic when the form is actually submitted
if ($_SERVER["REQUEST_METHOD"] !== "POST") {
    header("Location: index.php");
    exit();
}

// Now it's safe to read POST data
$email    = $_POST["email"]    ?? "";
$password = $_POST["password"] ?? "";

// Make sure neither field is blank
if ($email === "" || $password === "") {
    header("Location: index.php?error=missing");
    exit();
}

// Query the students table
$sql  = "SELECT * FROM students WHERE email = ?";
$stmt = $conn->prepare($sql);
$stmt->bind_param("s", $email);
$stmt->execute();
$result = $stmt->get_result();

if ($result->num_rows == 1) {
    $user = $result->fetch_assoc();

    if (password_verify($password, $user["password"])) {
        // Save session variables
        $_SESSION["user_id"]    = $user["id"];
        $_SESSION["name"]       = $user["first_name"];
        $_SESSION["student_id"] = $user["student_id"];
        $_SESSION["email"]      = $user["email"];

        header("Location: dashboard.php");
        exit();
    } else {
        header("Location: index.php?error=wrongpassword");
        exit();
    }
} else {
    header("Location: index.php?error=notfound");
    exit();
}
?>