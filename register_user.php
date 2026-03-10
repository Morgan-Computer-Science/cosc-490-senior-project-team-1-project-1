<?php

require_once "db.php";

/* GET FORM DATA */

$first = $_POST['first_name'];
$last = $_POST['last_name'];
$student = $_POST['student_id'];
$dob = $_POST['dob'];
$email = $_POST['email'];
$school_year = $_POST['school_year'];

/* REQUIRE MORGAN EMAIL */

if(!preg_match("/@morgan\.edu$/",$email)){
die("Only Morgan State University emails (@morgan.edu) are allowed.");
}

/* RESTRICT SCHOOL YEARS */

$allowedYears = ["Sophomore","Junior","Senior"];

if(!in_array($school_year,$allowedYears)){
die("Invalid school year selected");
}

/* CHECK IF EMAIL EXISTS */

$check = $conn->prepare("SELECT id FROM students WHERE email=?");
$check->bind_param("s",$email);
$check->execute();
$check->store_result();

if($check->num_rows > 0){
die("Account already exists. Please login.");
}

/* PASSWORD CHECK */

if($_POST['password'] !== $_POST['confirm_password']){
die("Passwords do not match");
}

/* HASH PASSWORD */

$password = password_hash($_POST['password'], PASSWORD_DEFAULT);

/* INSERT STUDENT */

$sql = "INSERT INTO students(first_name,last_name,student_id,dob,email,password,school_year)
VALUES(?,?,?,?,?,?,?)";

$stmt = $conn->prepare($sql);
$stmt->bind_param("sssssss",$first,$last,$student,$dob,$email,$password,$school_year);
$stmt->execute();

/* GET NEW STUDENT ID */

$newStudentId = $conn->insert_id;

/* AUTO GENERATE COMPLETED COURSES */

$courses = [];

if($school_year == "Sophomore"){
$courses = ["COSC100","COSC110","MATH115"];
}

if($school_year == "Junior"){
$courses = ["COSC100","COSC110","COSC111","COSC220","MATH115","MATH141"];
}

if($school_year == "Senior"){
$courses = ["COSC100","COSC110","COSC111","COSC220","COSC281","COSC320","MATH115","MATH141","MATH241"];
}

/* INSERT COMPLETED COURSES */

foreach($courses as $course){

$stmt = $conn->prepare("
INSERT INTO completed_courses(student_id,course_code)
VALUES(?,?)
");

$stmt->bind_param("is",$newStudentId,$course);
$stmt->execute();

}

/* REDIRECT */

header("Location: index.php?signup=success");
exit();

?>