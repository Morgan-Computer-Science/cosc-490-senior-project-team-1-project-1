<?php

session_start();
header("Content-Type: application/json");

require_once("db.php");
require_once "vendor/autoload.php";

use Smalot\PdfParser\Parser;

$apiKey = "YOUR_GEMINI_API_KEY";

/* ============================
   STUDENT SESSION INFO
============================ */

$name = $_SESSION["name"] ?? "Student";
$db_student_id = $_SESSION["user_id"] ?? "";
$student_number = $_SESSION["student_id"] ?? "";
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

$ch = curl_init($catalogURL);

curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_USERAGENT, "Mozilla/5.0");
curl_setopt($ch, CURLOPT_TIMEOUT, 5);

$catalogHTML = curl_exec($ch);
curl_close($ch);

$catalogPageText = "";

if($catalogHTML){

$catalogPageText = strip_tags($catalogHTML);
$catalogPageText = preg_replace('/\s+/', ' ', $catalogPageText);
$catalogPageText = substr($catalogPageText,0,5000);

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
   COURSE RECOMMENDATIONS
============================ */

$recommendedCourses=[];

foreach($catalog as $course){

$canTake=true;

foreach($completedCourses as $done){

if(strpos($course,$done)!==false){
$canTake=false;
}

}

if($canTake){
$recommendedCourses[]=$course;
}

}

$recommendedText = implode(", ",array_slice($recommendedCourses,0,10));


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

$url="https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key=".$apiKey;

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

$_SESSION["chat_history"][]=[
"role"=>"user",
"parts"=>[
["text"=>$message]
]
];

if(count($_SESSION["chat_history"])>10){
array_shift($_SESSION["chat_history"]);
}


/* ============================
   AI PROMPT
============================ */

$prompt="

You are Morgan AI, an academic advisor assistant for Morgan State University 🎓.

Your job is to help students:

1. Understand degree requirements
2. Recommend courses
3. Check prerequisites
4. Answer Morgan catalog questions

If database information conflicts with the official catalog,
always trust the Morgan catalog.

Response Style Rules:

Use short readable paragraphs.
Leave spacing between sections.
Use numbered lists for advice.
Keep responses friendly.
Use light emoji occasionally 📚 🎓 📅.

Student Information
Name: $name
Student ID: $student_number
Email: $email

Completed Courses
$completedText

Possible Courses They Can Take Next
$recommendedText

Morgan CS Curriculum Database
$catalogText

Official Morgan Catalog Website Data
$catalogPageText

Uploaded File Content
$fileText

Student Question
$message

";


/* ============================
   GEMINI REQUEST DATA
============================ */

$data=[
"contents"=>array_merge(
[
[
"role"=>"user",
"parts"=>[
["text"=>$prompt]
]
]
],
$_SESSION["chat_history"]
)
];


/* ============================
   GEMINI MODEL FALLBACK
============================ */

$models = [
"gemini-2.0-flash",
"gemini-1.5-flash"
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


if(isset($decoded["candidates"])){

$reply=$decoded["candidates"][0]["content"]["parts"][0]["text"] ?? "";


/* save response to history */

$_SESSION["chat_history"][]=[
"role"=>"model",
"parts"=>[
["text"=>$reply]
]
];


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


/* ============================
   IF ALL MODELS FAIL
============================ */

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

?>