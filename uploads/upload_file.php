<?php

$target="uploads/".basename($_FILES["file"]["name"]);

move_uploaded_file($_FILES["file"]["tmp_name"],$target);

echo "Uploaded";

?>