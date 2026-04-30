[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=22437268&assignment_repo_type=AssignmentRepo)

Morgan AI is a web-based application designed to help Morgan State University Computer Science students plan their academic journey. The platform allows students to communicate with an AI agent (Google Gemini) to get guidance on which classes to take each year in order to graduate.

Overview
Morgan AI helps students answer questions such as:
* What classes should I take next semester?
* Am I on track to graduate?
* What courses are required for my degree?
The system combines AI support with curriculum guidance to provide helpful academic planning.

Features
User Authentication
* Sign Up
    * New users can create an account
    * Only Morgan State University students are allowed
    * A valid Morgan email is required (no other emails are accepted)
* Login
    * Users log in with their created credentials
    * All user information is stored in the database

 Chat Page (AI Assistant)
* Students can communicate with an AI chatbot powered by Google Gemini
* Ask questions about:
    * Course selection
    * Degree requirements
    * Academic planning
* Users can also upload course curriculum files for better responses

Course Curriculum Page
* Accessed by clicking the Morgan AI logo
* Displays a visual of suggested classes for each academic year
* Helps students understand their path and prepare questions for the chatbot

 Profile Page
* Users can enter their expected graduation year
* This helps personalize recommendations

Setup
* The application runs locally using MAMP
* Uses a database named morgan_ai
* User data (such as sign-up information) is stored in the database

Application Access
Application Link: http://localhost:8888/morgan_ai/ 

 Target Users
* Morgan State University Computer Science students
* Students who want guidance on course planning and graduation

Access Restriction
* Only users with a valid Morgan State email can create an account
* Non-Morgan emails are not allowed

 Future Improvements
* More advanced AI recommendations
* Integration with official course data
* Improved user interface and mobile support
