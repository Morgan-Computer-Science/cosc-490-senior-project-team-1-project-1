-- phpMyAdmin SQL Dump
-- version 5.2.3
-- https://www.phpmyadmin.net/
--
-- Host: localhost:8889
-- Generation Time: Mar 10, 2026 at 09:22 PM
-- Server version: 8.0.44
-- PHP Version: 8.3.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `morgan_ai`
--

-- --------------------------------------------------------

--
-- Table structure for table `chat_messages`
--

CREATE TABLE `chat_messages` (
  `id` int NOT NULL,
  `student_id` int DEFAULT NULL,
  `sender` varchar(20) DEFAULT NULL,
  `message` text,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `completed_courses`
--

CREATE TABLE `completed_courses` (
  `id` int NOT NULL,
  `student_id` int DEFAULT NULL,
  `course_code` varchar(20) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `completed_courses`
--

INSERT INTO `completed_courses` (`id`, `student_id`, `course_code`) VALUES
(1, 2, 'COSC100'),
(2, 2, 'COSC110'),
(3, 2, 'COSC111'),
(4, 2, 'COSC220'),
(5, 2, 'COSC281'),
(6, 2, 'COSC320'),
(7, 2, 'MATH115'),
(8, 2, 'MATH141'),
(9, 2, 'MATH241'),
(10, 4, 'COSC100'),
(11, 4, 'COSC110'),
(12, 4, 'COSC111'),
(13, 4, 'COSC220'),
(14, 4, 'MATH115'),
(15, 4, 'MATH141'),
(16, 5, 'COSC100'),
(17, 5, 'COSC110'),
(18, 5, 'COSC111'),
(19, 5, 'COSC220'),
(20, 5, 'MATH115'),
(21, 5, 'MATH141'),
(22, 6, 'COSC100'),
(23, 6, 'COSC110'),
(24, 6, 'COSC111'),
(25, 6, 'COSC220'),
(26, 6, 'MATH115'),
(27, 6, 'MATH141'),
(28, 7, 'COSC100'),
(29, 7, 'COSC110'),
(30, 7, 'MATH115');

-- --------------------------------------------------------

--
-- Table structure for table `courses`
--

CREATE TABLE `courses` (
  `id` int NOT NULL,
  `course_code` varchar(20) DEFAULT NULL,
  `course_name` varchar(120) DEFAULT NULL,
  `credits` int DEFAULT NULL,
  `prerequisite` varchar(50) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `courses`
--

INSERT INTO `courses` (`id`, `course_code`, `course_name`, `credits`, `prerequisite`) VALUES
(1, 'COSC100', 'Introduction to Computer Science', 3, NULL),
(2, 'COSC110', 'Computer Programming I', 3, NULL),
(3, 'COSC111', 'Computer Programming II', 3, NULL),
(4, 'COSC220', 'Data Structures', 3, NULL),
(5, 'COSC281', 'Computer Organization', 3, NULL),
(6, 'COSC320', 'Algorithms', 3, NULL),
(7, 'COSC385', 'Software Engineering', 3, NULL),
(8, 'COSC490', 'Senior Design Project I', 3, NULL),
(9, 'COSC499', 'Senior Design Project II', 3, NULL),
(10, 'MATH115', 'College Algebra', 3, NULL),
(11, 'MATH141', 'Calculus I', 4, NULL),
(12, 'MATH241', 'Calculus II', 4, NULL),
(13, 'MATH313', 'Linear Algebra', 3, NULL);

-- --------------------------------------------------------

--
-- Table structure for table `settings`
--

CREATE TABLE `settings` (
  `id` int NOT NULL,
  `student_id` int DEFAULT NULL,
  `theme` varchar(20) DEFAULT 'dark',
  `notifications` tinyint(1) DEFAULT '1'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `students`
--

CREATE TABLE `students` (
  `id` int NOT NULL,
  `first_name` varchar(50) DEFAULT NULL,
  `last_name` varchar(50) DEFAULT NULL,
  `student_id` varchar(20) DEFAULT NULL,
  `dob` date DEFAULT NULL,
  `email` varchar(100) DEFAULT NULL,
  `password` varchar(255) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `school_year` varchar(20) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `students`
--

INSERT INTO `students` (`id`, `first_name`, `last_name`, `student_id`, `dob`, `email`, `password`, `created_at`, `school_year`) VALUES
(1, 'Jane', 'Doe', '00310148', '2004-03-05', 'jadoe1@morgan.edu', '$2y$10$Eml7p1mzbdHsrqFlF31QseVjmhkTvmvH.tdIw1ZY9sGBkfV8P2QNu', '2026-03-09 19:53:28', NULL),
(2, 'Micheal', 'Myers', '00310143', '2004-03-11', 'mkmyers1@morgan.edu', '$2y$10$Z1Bue1N9S5ChZA7FrBWE0.Yyh.JNZ0LLK0ErwH7oWJcWjRsRgP82W', '2026-03-10 19:45:05', 'Senior'),
(6, 'April', 'Koger', '00210132', '2005-02-18', 'akoger20@morgan.edu', '$2y$10$MtIUnTSocm00OZNeH.TVd.6xYJ3YBvP8cLr0jlUs5iKCNMgbMgowu', '2026-03-10 19:54:42', 'Junior'),
(7, 'Kevin', 'Hart', '00210432', '0004-03-10', 'khart1@morgan.edu', '$2y$10$CkthUE2Ya67Pnl3.eMyn3uNOX5XXsfrtj52uXrW73P1u81gyNdVJO', '2026-03-10 21:06:25', 'Sophomore');

-- --------------------------------------------------------

--
-- Table structure for table `support_messages`
--

CREATE TABLE `support_messages` (
  `id` int NOT NULL,
  `name` varchar(100) DEFAULT NULL,
  `email` varchar(150) DEFAULT NULL,
  `message` text,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `support_messages`
--

INSERT INTO `support_messages` (`id`, `name`, `email`, `message`, `created_at`) VALUES
(1, 'Jane Doe', 'jadoe1@morgan.edu', 'Hello just testing this', '2026-03-10 18:11:24');

-- --------------------------------------------------------

--
-- Table structure for table `support_requests`
--

CREATE TABLE `support_requests` (
  `id` int NOT NULL,
  `name` varchar(100) DEFAULT NULL,
  `email` varchar(100) DEFAULT NULL,
  `message` text,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `chat_messages`
--
ALTER TABLE `chat_messages`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `completed_courses`
--
ALTER TABLE `completed_courses`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `courses`
--
ALTER TABLE `courses`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `settings`
--
ALTER TABLE `settings`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `student_id` (`student_id`);

--
-- Indexes for table `students`
--
ALTER TABLE `students`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `student_id` (`student_id`),
  ADD UNIQUE KEY `email` (`email`);

--
-- Indexes for table `support_messages`
--
ALTER TABLE `support_messages`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `support_requests`
--
ALTER TABLE `support_requests`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `chat_messages`
--
ALTER TABLE `chat_messages`
  MODIFY `id` int NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `completed_courses`
--
ALTER TABLE `completed_courses`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=31;

--
-- AUTO_INCREMENT for table `courses`
--
ALTER TABLE `courses`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=14;

--
-- AUTO_INCREMENT for table `settings`
--
ALTER TABLE `settings`
  MODIFY `id` int NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `students`
--
ALTER TABLE `students`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT for table `support_messages`
--
ALTER TABLE `support_messages`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `support_requests`
--
ALTER TABLE `support_requests`
  MODIFY `id` int NOT NULL AUTO_INCREMENT;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
