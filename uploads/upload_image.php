<?php

$target="uploads/".basename($_FILES["image"]["name"]);

move_uploaded_file($_FILES["image"]["tmp_name"],$target);

echo "Uploaded";

?>