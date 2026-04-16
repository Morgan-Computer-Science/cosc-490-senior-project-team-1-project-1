<?php

session_start();
header("Content-Type: application/json");

require_once("db.php");
require_once "vendor/autoload.php";

use Smalot\PdfParser\Parser;
$apiKey = "AIzaSyCkTC-KxaFLF4q5MmKKYQ2IX7EwXqssSzg";

if (!isset($_SESSION["user_id"])) {
    echo json_encode([
        "candidates" => [[
            "content" => ["parts" => [["text" => "Session expired. Please log in again."]]]
        ]]
    ]);
    exit();
}

$name           = $_SESSION["name"]       ?? "Student";
$db_student_id  = $_SESSION["user_id"]    ?? "";
$student_number = $_SESSION["student_id"] ?? "";
$email          = $_SESSION["email"]      ?? "";

$completedCourses = [];

if ($db_student_id) {
    $courseQuery = $conn->prepare("
        SELECT course_code
        FROM completed_courses
        WHERE student_id = ?
    ");
    $courseQuery->bind_param("i", $db_student_id);
    $courseQuery->execute();
    $result = $courseQuery->get_result();
    while ($row = $result->fetch_assoc()) {
        $completedCourses[] = $row["course_code"];
    }
}

$completedText = !empty($completedCourses)
    ? implode(", ", $completedCourses)
    : "No completed courses on record yet.";


if (!isset($_SESSION["catalog_cache"])) {
    $catalogURL = "https://catalog.morgan.edu/preview_program.php?catoid=26&poid=5968&returnto=1880";
    $ch = curl_init($catalogURL);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_USERAGENT, "Mozilla/5.0");
    curl_setopt($ch, CURLOPT_TIMEOUT, 5);
    $catalogHTML = curl_exec($ch);
    curl_close($ch);

    if ($catalogHTML) {
        $text = strip_tags($catalogHTML);
        $text = preg_replace('/\s+/', ' ', $text);
        $_SESSION["catalog_cache"] = substr($text, 0, 5000);
    } else {
        $_SESSION["catalog_cache"] = "";
    }
}

$catalogPageText = $_SESSION["catalog_cache"];
$catalog = [];

$result = $conn->query("
    SELECT course_code, course_name, prerequisite
    FROM courses
");

while ($row = $result->fetch_assoc()) {
    $catalog[] =
        $row["course_code"] . " " . $row["course_name"] .
        " (Prerequisite: " . ($row["prerequisite"] ?? "None") . ")";
}

$catalogText = !empty($catalog)
    ? implode(", ", $catalog)
    : "No curriculum data found.";

$recommendedCourses = [];

foreach ($catalog as $course) {
    $canTake = true;
    foreach ($completedCourses as $done) {
        if (strpos($course, $done) !== false) {
            $canTake = false;
            break;
        }
    }
    if ($canTake) {
        $recommendedCourses[] = $course;
    }
}

$recommendedText = !empty($recommendedCourses)
    ? implode(", ", array_slice($recommendedCourses, 0, 10))
    : "All available courses may have been completed.";

$message = isset($_POST["message"]) ? trim($_POST["message"]) : "";

if ($message === "" && (!isset($_FILES["file"]) || $_FILES["file"]["error"] !== UPLOAD_ERR_OK)) {
    echo json_encode([
        "candidates" => [[
            "content" => ["parts" => [["text" => "I didn't receive a message. Please type something and try again."]]]
        ]]
    ]);
    exit();
}

$fileText = "";

if (isset($_FILES["file"]) && $_FILES["file"]["error"] === UPLOAD_ERR_OK) {

    $fileTmp  = $_FILES["file"]["tmp_name"];
    $fileName = $_FILES["file"]["name"];
    $fileType = strtolower(pathinfo($fileName, PATHINFO_EXTENSION));
    if ($fileType === "pdf") {
        try {
            $parser   = new Parser();
            $pdf      = $parser->parseFile($fileTmp);
            $fileText = $pdf->getText();
        } catch (Exception $e) {
            $fileText = "Could not read PDF: " . $e->getMessage();
        }
    }

    elseif ($fileType === "txt") {
        $fileText = file_get_contents($fileTmp);
    }

    elseif ($fileType === "docx") {
        $zip = new ZipArchive;
        if ($zip->open($fileTmp) === TRUE) {
            $xml      = $zip->getFromName("word/document.xml");
            $zip->close();
            $fileText = strip_tags($xml);
        }
    }

    elseif (in_array($fileType, ["jpg", "jpeg", "png"])) {

        $imageData = base64_encode(file_get_contents($fileTmp));

        $visionURL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key=" . $apiKey;

        $visionPayload = [
            "contents" => [[
                "parts" => [
                    ["text" => "Describe the academic information shown in this image."],
                    ["inline_data" => [
                        "mime_type" => "image/" . ($fileType === "jpg" ? "jpeg" : $fileType),
                        "data"      => $imageData
                    ]]
                ]
            ]]
        ];

        $ch = curl_init($visionURL);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_HTTPHEADER, ["Content-Type: application/json"]);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($visionPayload));
        curl_setopt($ch, CURLOPT_TIMEOUT, 15);
        $visionResponse = curl_exec($ch);
        curl_close($ch);

        $visionDecoded = json_decode($visionResponse, true);
        $fileText      = $visionDecoded["candidates"][0]["content"]["parts"][0]["text"] ?? "Could not read image.";
    }
}



if (!isset($_SESSION["chat_history"])) {
    $_SESSION["chat_history"] = [];
}

$_SESSION["chat_history"][] = [
    "role"  => "user",
    "parts" => [["text" => $message]]
];

if (count($_SESSION["chat_history"]) > 10) {
    array_shift($_SESSION["chat_history"]);
}


$systemPrompt = "
You are Morgan AI, a friendly and knowledgeable academic advisor assistant
for Morgan State University students 🎓.

Your job is to help students with:
1. Understanding their degree requirements
2. Recommending courses based on what they have completed
3. Checking prerequisites before suggesting courses
4. Answering questions about the Morgan State course catalog
5. General academic advice and campus resources

IMPORTANT RULES:
- If the database conflicts with the official Morgan catalog, trust the Morgan catalog.
- Never recommend a course whose prerequisites the student has not completed.
- Be encouraging and supportive.
- If unsure about something specific to Morgan State, say so and direct the student
  to their academic advisor or morgan.edu.
- Do NOT use markdown bold (** **) in your responses — write in plain text only.

Response Style:
- Short, readable paragraphs with spacing between sections.
- Numbered lists for step-by-step advice.
- Friendly, warm, professional tone.
- Light emoji occasionally 📚 🎓 📅 ✅ — but don't overdo it.

--- STUDENT INFORMATION ---
Name: $name
Student ID: $student_number
Email: $email

--- COMPLETED COURSES ---
$completedText

--- RECOMMENDED NEXT COURSES (based on completed courses + prerequisites) ---
$recommendedText

--- FULL MORGAN CS CURRICULUM (from database) ---
$catalogText

--- OFFICIAL MORGAN CATALOG (live website snapshot) ---
$catalogPageText

--- UPLOADED FILE CONTENT (if any) ---
" . ($fileText !== "" ? $fileText : "No file uploaded this message.") . "

--- STUDENT'S QUESTION ---
$message
";


$contents = array_merge(
    [[
        "role"  => "user",
        "parts" => [["text" => $systemPrompt]]
    ]],
    $_SESSION["chat_history"]
);

$requestData = ["contents" => $contents];



$models = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
];

foreach ($models as $model) {

    $url = "https://generativelanguage.googleapis.com/v1/models/" . $model . ":generateContent?key=" . $apiKey;

    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, ["Content-Type: application/json"]);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($requestData));
    curl_setopt($ch, CURLOPT_TIMEOUT, 20);

    $response = curl_exec($ch);
    $curlErr  = curl_error($ch);
    curl_close($ch);

    if ($curlErr) {
        error_log("Morgan AI cURL error ($model): $curlErr");
        continue;
    }

    $decoded = json_decode($response, true);

    if (isset($decoded["candidates"][0]["content"]["parts"][0]["text"])) {

        $reply = $decoded["candidates"][0]["content"]["parts"][0]["text"];

        $_SESSION["chat_history"][] = [
            "role"  => "model",
            "parts" => [["text" => $reply]]
        ];

        echo json_encode([
            "candidates" => [[
                "content" => [
                    "parts" => [["text" => $reply]]
                ]
            ]]
        ]);
        exit();
    }

    $errorMsg = $decoded["error"]["message"] ?? json_encode($decoded);
    error_log("Morgan AI: $model failed — $errorMsg");
}

echo json_encode([
    "candidates" => [[
        "content" => [
            "parts" => [["text" => "Morgan AI is temporarily busy. Please try again in a moment. ⏳"]]
        ]
    ]]
]);
exit();