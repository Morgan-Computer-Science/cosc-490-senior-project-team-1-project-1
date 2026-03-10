<?php

session_start();
header("Content-Type: application/json");

require_once("db.php");
require_once "vendor/autoload.php";

use Smalot\PdfParser\Parser;

$apiKey = "AIzaSyBirH_IC08w6Yj59xd6YxRC0xfYchZAz88";

/* ============================
   STUDENT SESSION INFO
============================ */

$name = $_SESSION["name"] ?? "Student";

$db_student_id = $_SESSION["user_id"] ?? "";      // database id
$student_number = $_SESSION["student_id"] ?? "";  // real Morgan student number

$email = $_SESSION["email"] ?? "";

/* ============================
   GET COMPLETED COURSES
============================ */

$completedCourses = [];

if($db_student_id){

$courseQuery = $conn->prepare("
SELECT course_code
FROM completed_courses
WHERE student_id=?
");

$courseQuery->bind_param("i",$db_student_id);
$courseQuery->execute();

$result = $courseQuery->get_result();

while($row=$result->fetch_assoc()){
$completedCourses[] = $row["course_code"];
}

}

$completedText = implode(", ",$completedCourses);

/* ============================
   GET MORGAN CATALOG (RAG)
============================ */

$catalogURL = "https://catalog.morgan.edu/preview_program.php?catoid=26&poid=5968&returnto=1880";

$catalogHTML = @file_get_contents($catalogURL, false, stream_context_create([
    "http" => [
        "timeout" => 3
    ]
]));

$catalogPageText = "";

if($catalogHTML){

$catalogPageText = strip_tags($catalogHTML);

/* reduce size so it fits AI prompt */

$catalogPageText = substr($catalogPageText,0,8000);

}

/* ============================
   GET CURRICULUM FROM DATABASE
============================ */

$catalog = [];

$result = $conn->query("
SELECT course_code, course_name, prerequisite
FROM courses
");

while($row = $result->fetch_assoc()){

$catalog[] =
$row["course_code"] .
" " .
$row["course_name"] .
" (Prerequisite: " .
($row["prerequisite"] ?? "None") .
")";

}

$catalogText = implode(", ", $catalog);

/* ============================
   MESSAGE INPUT
============================ */

$message = $_POST["message"] ?? "";

/* ============================
   FILE PROCESSING
============================ */

$fileText="";

if(isset($_FILES["file"])){

$fileTmp=$_FILES["file"]["tmp_name"];
$fileName=$_FILES["file"]["name"];

$fileType=strtolower(pathinfo($fileName, PATHINFO_EXTENSION));

/* PDF */

if($fileType=="pdf"){

$parser=new Parser();
$pdf=$parser->parseFile($fileTmp);
$fileText=$pdf->getText();

}

/* TXT */

elseif($fileType=="txt"){
$fileText=file_get_contents($fileTmp);
}

/* DOCX */

elseif($fileType=="docx"){

$zip=new ZipArchive;

if($zip->open($fileTmp)===TRUE){

$xml=$zip->getFromName("word/document.xml");

$zip->close();

$fileText=strip_tags($xml);

}

}

/* IMAGE */

elseif(in_array($fileType,["jpg","jpeg","png"])){

$imageData=base64_encode(file_get_contents($fileTmp));

$url="https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key=".$apiKey;

$data=[
"contents"=>[
[
"parts"=>[
["text"=>"Describe the academic information shown in this image."],
[
"inline_data"=>[
"mime_type"=>"image/".$fileType,
"data"=>$imageData
]
]
]
]
]
];

$ch=curl_init($url);

curl_setopt($ch,CURLOPT_RETURNTRANSFER,true);
curl_setopt($ch,CURLOPT_POST,true);
curl_setopt($ch,CURLOPT_HTTPHEADER,["Content-Type: application/json"]);
curl_setopt($ch,CURLOPT_POSTFIELDS,json_encode($data));

$visionResponse=curl_exec($ch);

curl_close($ch);

$decoded=json_decode($visionResponse,true);

$fileText=$decoded["candidates"][0]["content"]["parts"][0]["text"] ?? "";

}

}

/* ============================
   CHAT HISTORY
============================ */

if(!isset($_SESSION["chat_history"])){
$_SESSION["chat_history"]=[];
}

$_SESSION["chat_history"][]="Student: ".$message;

if(count($_SESSION["chat_history"])>8){
array_shift($_SESSION["chat_history"]);
}

$history=implode("\n",$_SESSION["chat_history"]);

/* ============================
   AI PROMPT
============================ */

$prompt="
Always check the prerequisite listed for each course before advising whether a student can enroll.

You are Morgan AI, an academic advisor assistant for Morgan State University.

If database information conflicts with the official Morgan catalog,
always trust the catalog.

RESPONSE STYLE RULES:

Do not use markdown symbols like ** or *.
Use short readable paragraphs.
Leave spacing between sections.
Use numbered lists for advice.
Keep responses conversational.
Use light emoji occasionally (📚 🎓 📅).

Student Information:
Name: $name
Student ID: $student_number
Email: $email

Completed Courses:
$completedText

Morgan Computer Science Curriculum:
$catalogText

Official Morgan State Catalog Information:
$catalogPageText

Conversation history:
$history

Uploaded file content:
$fileText

Student question:
$message

";

/* ============================
   GEMINI REQUEST DATA
============================ */

$data=[
"contents"=>[
[
"parts"=>[
["text"=>$prompt]
]
]
]
];

/* ============================
   GEMINI MODEL FALLBACK
============================ */

$models = [
"gemini-2.0-flash",   // free + most reliable
"gemini-2.5-flash",   // faster but limited quota
"gemini-1.5-flash"    // extra backup
];

foreach($models as $model){

$url="https://generativelanguage.googleapis.com/v1/models/".$model.":generateContent?key=".$apiKey;

$ch=curl_init($url);

curl_setopt($ch,CURLOPT_RETURNTRANSFER,true);
curl_setopt($ch,CURLOPT_POST,true);
curl_setopt($ch,CURLOPT_HTTPHEADER,["Content-Type: application/json"]);
curl_setopt($ch,CURLOPT_POSTFIELDS,json_encode($data));

$response=curl_exec($ch);
curl_close($ch);

$decoded=json_decode($response,true);

/* SUCCESS */

if(isset($decoded["candidates"])){

$reply=$decoded["candidates"][0]["content"]["parts"][0]["text"] ?? "";

$_SESSION["chat_history"][]="Morgan AI: ".$reply;

echo json_encode([
"candidates"=>[
[
"content"=>[
"parts"=>[
["text"=>$reply]
]
]
]
]
]);

exit();

}

}

/* IF ALL MODELS FAIL */

echo json_encode([
"candidates"=>[
[
"content"=>[
"parts"=>[
["text"=>"Morgan AI is temporarily busy. Please try again in a moment."]
]
]
]
]
]);