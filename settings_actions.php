<?php
session_start();
require_once("db.php");

header("Content-Type: application/json");

if(!isset($_SESSION["user_id"])){
    http_response_code(401);
    echo json_encode(["ok"=>false,"message"=>"Unauthorized"]);
    exit();
}

$studentId = (int)$_SESSION["user_id"];
$action = $_POST["action"] ?? "";

function respond(bool $ok,string $message,array $data=[]): void {
    echo json_encode(array_merge(["ok"=>$ok,"message"=>$message],$data));
    exit();
}

function ensureSettingsRows(mysqli $conn,int $studentId): void {
    $conn->query("CREATE TABLE IF NOT EXISTS student_academic_settings (
        id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        student_id INT NOT NULL UNIQUE,
        graduation VARCHAR(50) DEFAULT NULL,
        assigned_advisor VARCHAR(100) DEFAULT NULL,
        degree_track VARCHAR(100) DEFAULT NULL,
        created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4");

    $stmt = $conn->prepare("INSERT INTO settings (student_id) VALUES (?) ON DUPLICATE KEY UPDATE student_id=VALUES(student_id)");
    if($stmt){
        $stmt->bind_param("i",$studentId);
        $stmt->execute();
    }

    $stmt2 = $conn->prepare("INSERT INTO student_academic_settings (student_id) VALUES (?) ON DUPLICATE KEY UPDATE student_id=VALUES(student_id)");
    if($stmt2){
        $stmt2->bind_param("i",$studentId);
        $stmt2->execute();
    }
}

ensureSettingsRows($conn,$studentId);

if($action === "load_settings"){
    $profileStmt = $conn->prepare("SELECT first_name,last_name,email,student_id FROM students WHERE id=?");
    $profileStmt->bind_param("i",$studentId);
    $profileStmt->execute();
    $profile = $profileStmt->get_result()->fetch_assoc();

    $settingsStmt = $conn->prepare("SELECT theme,notifications FROM settings WHERE student_id=?");
    $settingsStmt->bind_param("i",$studentId);
    $settingsStmt->execute();
    $settings = $settingsStmt->get_result()->fetch_assoc();

    $academicStmt = $conn->prepare("SELECT graduation,assigned_advisor,degree_track FROM student_academic_settings WHERE student_id=?");
    $academicStmt->bind_param("i",$studentId);
    $academicStmt->execute();
    $academic = $academicStmt->get_result()->fetch_assoc();

    respond(true,"Loaded",[
        "profile"=>$profile ?: [],
        "settings"=>$settings ?: ["theme"=>"dark","notifications"=>1],
        "academic"=>$academic ?: []
    ]);
}

if($action === "edit_profile"){
    $first = trim($_POST["first_name"] ?? "");
    $last = trim($_POST["last_name"] ?? "");
    $studentCode = trim($_POST["student_id"] ?? "");

    if($first === "" || $last === "" || $studentCode === ""){
        respond(false,"All profile fields are required.");
    }

    $check = $conn->prepare("SELECT id FROM students WHERE student_id=? AND id<>?");
    $check->bind_param("si",$studentCode,$studentId);
    $check->execute();
    $check->store_result();
    if($check->num_rows > 0){
        respond(false,"That Student ID is already in use.");
    }

    $stmt = $conn->prepare("UPDATE students SET first_name=?,last_name=?,student_id=? WHERE id=?");
    $stmt->bind_param("sssi",$first,$last,$studentCode,$studentId);
    $stmt->execute();
    respond(true,"Profile updated.");
}

if($action === "change_password"){
    $current = $_POST["current_password"] ?? "";
    $new = $_POST["new_password"] ?? "";
    $confirm = $_POST["confirm_password"] ?? "";

    if($current === "" || $new === "" || $confirm === ""){
        respond(false,"All password fields are required.");
    }
    if($new !== $confirm){
        respond(false,"New password and confirmation do not match.");
    }
    if(strlen($new) < 8){
        respond(false,"New password must be at least 8 characters.");
    }

    $stmt = $conn->prepare("SELECT password FROM students WHERE id=?");
    $stmt->bind_param("i",$studentId);
    $stmt->execute();
    $row = $stmt->get_result()->fetch_assoc();
    if(!$row || !password_verify($current,$row["password"])){
        respond(false,"Current password is incorrect.");
    }

    $hash = password_hash($new,PASSWORD_DEFAULT);
    $update = $conn->prepare("UPDATE students SET password=? WHERE id=?");
    $update->bind_param("si",$hash,$studentId);
    $update->execute();
    respond(true,"Password updated.");
}

if($action === "update_email"){
    $email = strtolower(trim($_POST["email"] ?? ""));
    if($email === ""){
        respond(false,"Email is required.");
    }
    if(!preg_match("/@morgan\\.edu$/",$email)){
        respond(false,"Email must end with @morgan.edu.");
    }

    $check = $conn->prepare("SELECT id FROM students WHERE email=? AND id<>?");
    $check->bind_param("si",$email,$studentId);
    $check->execute();
    $check->store_result();
    if($check->num_rows > 0){
        respond(false,"That email is already in use.");
    }

    $stmt = $conn->prepare("UPDATE students SET email=? WHERE id=?");
    $stmt->bind_param("si",$email,$studentId);
    $stmt->execute();
    respond(true,"Email updated.");
}

if($action === "preference"){
    $theme = trim($_POST["theme"] ?? "");
    $allowed = ["dark","light","system"];
    if(!in_array($theme,$allowed,true)){
        respond(false,"Preference must be one of: dark, light, system.");
    }

    $stmt = $conn->prepare("UPDATE settings SET theme=? WHERE student_id=?");
    $stmt->bind_param("si",$theme,$studentId);
    $stmt->execute();
    respond(true,"Preference updated.");
}

if($action === "notifications"){
    $enabled = ($_POST["enabled"] ?? "0") === "1" ? 1 : 0;
    $stmt = $conn->prepare("UPDATE settings SET notifications=? WHERE student_id=?");
    $stmt->bind_param("ii",$enabled,$studentId);
    $stmt->execute();
    respond(true,"Notification setting updated.");
}

if($action === "graduation"){
    $graduation = trim($_POST["graduation"] ?? "");
    if($graduation === ""){
        respond(false,"Graduation value is required.");
    }
    if(strlen($graduation) > 50){
        respond(false,"Graduation value is too long.");
    }
    $stmt = $conn->prepare("UPDATE student_academic_settings SET graduation=? WHERE student_id=?");
    $stmt->bind_param("si",$graduation,$studentId);
    $stmt->execute();
    respond(true,"Graduation updated.");
}

if($action === "assigned_advisor"){
    $advisor = trim($_POST["advisor"] ?? "");
    if($advisor === ""){
        respond(false,"Advisor name is required.");
    }
    if(strlen($advisor) > 100){
        respond(false,"Advisor name is too long.");
    }
    $stmt = $conn->prepare("UPDATE student_academic_settings SET assigned_advisor=? WHERE student_id=?");
    $stmt->bind_param("si",$advisor,$studentId);
    $stmt->execute();
    respond(true,"Assigned advisor updated.");
}

if($action === "degree_track"){
    $track = trim($_POST["degree_track"] ?? "");
    if($track === ""){
        respond(false,"Degree track is required.");
    }
    if(strlen($track) > 100){
        respond(false,"Degree track is too long.");
    }
    $stmt = $conn->prepare("UPDATE student_academic_settings SET degree_track=? WHERE student_id=?");
    $stmt->bind_param("si",$track,$studentId);
    $stmt->execute();
    respond(true,"Degree track updated.");
}

respond(false,"Unknown action.");
?>
