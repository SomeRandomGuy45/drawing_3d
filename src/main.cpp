#include "helper.h"
#include <atomic>
#include <mutex>

std::atomic<bool> windowReady(false);
std::condition_variable cv;
std::mutex mtx;

GLFWwindow* window;

bool locked = true;

std::vector<std::tuple<GLuint, std::pair<unsigned int, glm::vec3>, std::tuple<glm::vec3, glm::vec3, glm::vec3>>> models;

// Camera class
class Camera {
public:
    Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
        : Position(position), WorldUp(up), Yaw(yaw), Pitch(pitch) {
        updateCameraVectors();
    }

    glm::mat4 getViewMatrix() const {
        return glm::lookAt(Position, Position + Front, Up);
    }

    void processKeyboard(float deltaTime) {
        const float cameraSpeed = 2.5f * deltaTime;
        glm::vec3 resetC = Front;
        glm::vec3 resetUP = Up;
        Up.x = 0.0f;
        Up.z = 0.0f;
        Front.y = 0.0f;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            Position += Front * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            Position -= Front * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            Position -= Right * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            Position += Right * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            Position += Up * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
            Position -= Up * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            auto CURSOR_TYPE = locked ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED;
            glfwSetInputMode(window, GLFW_CURSOR, CURSOR_TYPE);
            locked = locked ? false : true;
        }
        Front = resetC;
        Up = resetUP;
    }

    void processMouseMovement(float xoffset, float yoffset) {
        const float sensitivity = 0.1f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        Yaw += xoffset;
        Pitch -= yoffset;

        if (Pitch > 89.0f) Pitch = 89.0f;
        if (Pitch < -89.0f) Pitch = -89.0f;

        updateCameraVectors();
    }

private:
    void updateCameraVectors() {
        glm::vec3 front;
        front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        front.y = sin(glm::radians(Pitch));
        front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        Front = glm::normalize(front);
        Right = glm::normalize(glm::cross(Front, WorldUp));
        Up = glm::normalize(glm::cross(Right, Front));
    }

    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    float Yaw;
    float Pitch;
};

// Function to load an OBJ file using Assimp
std::tuple<GLuint, std::pair<unsigned int, glm::vec3>, std::tuple<glm::vec3, glm::vec3, glm::vec3>> loadModel(const std::string& path, glm::vec3 position, glm::vec3 color, glm::vec3 scale, glm::vec3 rotationAxis) {
    std::vector<GLuint> indices;
    std::vector<glm::vec3> vertices;

    color.x /= 255.0f;
    color.y /= 255.0f;
    color.z /= 255.0f;

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
        return { 0, { 0, glm::vec3(0.0f) }, { glm::vec3(1.0f), glm::vec3(1.0f), glm::vec3(0.0f) } }; 
    }

    for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[i];
        for (unsigned int j = 0; j < mesh->mNumVertices; j++) {
            aiVector3D pos = mesh->mVertices[j];
            vertices.emplace_back(pos.x, pos.y, pos.z);
        }
        for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
            aiFace face = mesh->mFaces[j];
            for (unsigned int k = 0; k < face.mNumIndices; k++) {
                indices.push_back(face.mIndices[k]);
            }
        }
    }

    std::cout << "Loaded " << vertices.size() << " vertices and " << indices.size() << " indices." << std::endl;

    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return { VAO, { static_cast<unsigned int>(indices.size()), position }, { color, scale, rotationAxis } };
}

// Function to render all loaded models
void renderModels(GLuint shaderProgram) {
    for (const auto& model : models) {
        GLuint VAO = std::get<0>(model);
        unsigned int indexCount = std::get<1>(model).first;
        glm::vec3 position = std::get<1>(model).second;
        glm::vec3 color = std::get<0>(std::get<2>(model));
        glm::vec3 scale = std::get<1>(std::get<2>(model));
        glm::vec3 rotationAxis = std::get<2>(std::get<2>(model));

        // Normalize the axis and calculate the angle
        float rotationAngle = glm::length(rotationAxis);
        if (rotationAngle > 0.0f) {
            rotationAxis = glm::normalize(rotationAxis);
        }

        glm::mat4 modelMatrix = glm::translate(glm::mat4(1.0f), position);
        modelMatrix = glm::rotate(modelMatrix, rotationAngle, rotationAxis); // Rotation
        modelMatrix = glm::scale(modelMatrix, scale); // Scaling
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(modelMatrix));

        // Set color for the fragment shader
        glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, glm::value_ptr(color));

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}

//deltaTime
float currentDeltaTime;

// Tween parameters
struct Tween {
    glm::vec3 startRotation; // Starting rotation (in degrees)
    glm::vec3 endRotation;   // Target rotation (in degrees)
    float duration;          // Duration of the tween in seconds
    float elapsedTime;       // Time elapsed since the tween started
    float speed; // Speed of the tween
};

// Map to hold tween data for models
std::map<int, Tween> needToTween;
std::map<int, Tween> needToTween_POS;

float LinearEase(float t) {
    return t; // Linear interpolation
}

float EaseIn(float t) {
    return t * t; // Accelerating from zero velocity
}

float EaseOut(float t) {
    return t * (2 - t); // Decelerating to zero velocity
}

float EaseInOut(float t) {
    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; // Acceleration until halfway, then deceleration
}

glm::vec3 DoTweenFunc(const glm::vec3& start, const glm::vec3& end, float duration, float (*easeFunc)(float)) {
    float t = currentDeltaTime / duration; // Calculate the progress
    t = glm::clamp(t, 0.0f, 1.0f); // Clamp to [0, 1]
    
    // Apply the easing function
    float easedT = easeFunc(t);
    
    // Interpolate
    return start + (end - start) * easedT; 
}

// Vertex Shader
const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 position;

out vec3 fragColor; // Color passed to fragment shader

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    fragColor = vec3(1.0); // Default fragment color (white)
}
)";

// Fragment Shader
const char* fragmentShaderSource = R"(
#version 330 core
in vec3 fragColor;
out vec4 outColor;

uniform vec3 color;

void main() {
    outColor = vec4(color, 1.0); // Output the fragment color
}
)";

// Function to compile shaders and create a shader program
void checkCompileErrors(GLuint shader, const std::string& type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
                      << infoLog << "\n -- --------------------------------------------------- -- "
                      << std::endl;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n"
                      << infoLog << "\n -- --------------------------------------------------- -- "
                      << std::endl;
        }
    }
}

GLuint compileShaders() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    checkCompileErrors(vertexShader, "VERTEX");

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    checkCompileErrors(fragmentShader, "FRAGMENT");

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    checkCompileErrors(shaderProgram, "PROGRAM");

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

// Mouse callback to capture mouse movement for camera control
void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (!locked) return;
    static float lastX = 400, lastY = 300;
    static bool firstMouse = true;

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = ypos - lastY;

    Camera* camera = static_cast<Camera*>(glfwGetWindowUserPointer(window));
    camera->processMouseMovement(xoffset, yoffset);

    lastX = xpos;
    lastY = ypos;
}

void RemoveModel(int index)
{
    auto model = models[index];
    GLuint VAO = std::get<0>(model); // Extract the VAO from the model tuple
    glDeleteVertexArrays(1, &VAO);
    models.erase(std::remove(models.begin(), models.end(), model), models.end());
}

void MoveModel(int index, glm::vec3 newPosition, bool tween, float duration)
{
    if (index < 0 || index >= models.size())
    {
        std::cerr << "Invalid model index: " << index << std::endl;
        return;
    }

    auto& model = models[index];
    glm::vec3 currentPosition = std::get<1>(model).second;
    glm::vec3 direction = newPosition - currentPosition;
    float totalDistance = glm::length(direction);
    if (tween)
    {
        needToTween_POS[index] = {currentPosition, newPosition, duration, 0.0f, totalDistance/duration};
    }
    else
    {
        std::get<1>(model).second = newPosition;
    }
}

void MoveModel(int index, glm::vec3 newPosition) {MoveModel(index, newPosition, false, 0);};
void MoveModel(int index, glm::vec3 newPosition, bool tween) {MoveModel(index, newPosition, tween, 0);};

void RotateModel(int index, glm::vec3 newRotation, bool tween, float duration)
{
    if (index < 0 || index >= models.size()) {
        std::cerr << "Invalid model index: " << index << std::endl;
        return;
    }

    // Get the current model's rotation
    auto& model = models[index];
    glm::vec3 currentRotation = std::get<2>(std::get<2>(model)); // Assuming the rotation is stored here
    glm::vec3 direction = newRotation - currentRotation;
    float totalDistance = glm::length(direction);

    if (tween) {
        // Initialize tweening parameters
        needToTween[index] = {currentRotation, newRotation, duration, 0.0f, totalDistance};
    } else {
        // Directly set the new rotation if not tweening
        std::get<2>(std::get<2>(model)) = newRotation;
    }
}

void RotateModel(int index, glm::vec3 newRotation) {RotateModel(index, newRotation, false, 0);};
void RotateModel(int index, glm::vec3 newRotation, bool tween) {RotateModel(index, newRotation, tween, 0);};

void DoAllTweenRotate()
{
    for (auto it = needToTween.begin(); it != needToTween.end(); ) {
        auto& [index, tween] = *it;

        // If this is the first update, calculate the angular speed
        if (tween.elapsedTime == 0.0f) {
            // Calculate the angular distance to rotate
            glm::vec3 direction = tween.endRotation - tween.startRotation;
            float totalAngle = glm::length(direction); // Total angular distance
            tween.speed = totalAngle / tween.duration; // Compute angular speed
        }

        // Update elapsed time
        tween.elapsedTime += currentDeltaTime;

        // Calculate the angular distance to rotate this frame
        float angleToRotate = tween.speed * currentDeltaTime;

        // Calculate the current rotation
        glm::vec3 currentRotation = std::get<2>(std::get<2>(models[index]));

        // Determine if we can reach the target in this frame
        if (glm::length(tween.endRotation - currentRotation) > angleToRotate) {
            // Move towards the target by the calculated angle
            glm::vec3 rotationStep = glm::normalize(tween.endRotation - currentRotation) * angleToRotate;
            currentRotation += rotationStep; // Update current rotation
        } else {
            // Snap to the target rotation when within the distance threshold
            currentRotation = tween.endRotation;
            std::cout << "Tween complete for model " << index << std::endl;
            it = needToTween.erase(it); // Remove the completed tween
            continue; // Skip the increment as we've erased the element
        }

        // Update model rotation
        auto& model = models[index];
        std::get<2>(std::get<2>(model)) = currentRotation;

        // Debug output
        std::cout << "Current Rotation for model " << index << ": " << glm::to_string(currentRotation) << std::endl;

        ++it; // Move to the next tween
    }
}

void DoAllTweenMove()
{
    for (auto it = needToTween_POS.begin(); it != needToTween_POS.end(); ) {
        auto& [index, tween] = *it;

        // Get the current position
        glm::vec3 currentPos = std::get<1>(models[index]).second;

        // Calculate the total distance to the target position
        glm::vec3 direction = tween.endRotation - tween.startRotation;
        float totalDistance = glm::length(direction);

        // If speed is not already set, calculate it based on duration
        if (tween.elapsedTime == 0.0f) {
            tween.speed = totalDistance / tween.duration; // Compute speed
        }

        // Normalize the direction
        direction = glm::normalize(direction);

        // Calculate the distance to move this frame
        float distanceToMove = tween.speed * currentDeltaTime;

        // Check if we can reach the target in this frame
        if (glm::distance(currentPos, tween.endRotation) > distanceToMove) {
            // Move the object a fixed distance in the direction of the target
            currentPos += direction * distanceToMove;
        } else {
            // Snap to the target position when within the distance threshold
            currentPos = tween.endRotation; 
            std::cout << "Tween complete for model " << index << std::endl;
            it = needToTween_POS.erase(it); // Remove the completed tween
            continue; // Skip the increment, as we have erased the element
        }

        // Update model position
        std::get<1>(models[index]).second = currentPos;

        // Update elapsed time
        tween.elapsedTime += currentDeltaTime;

        ++it; // Move to the next tween
    }
}

void CreateModel_Test()
{
    models.push_back(loadModel("test.obj", glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(255.0f,0.0f,0.0f), glm::vec3(1.0f), glm::vec3(90.0f, 45.0f, 90.0f)));
}

void ClearColor()
{
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
}

int createNewWindow() {
    std::cout << "Creating new window\n";

    // Initialize GLFW
    glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
    glfwInitHint(GLFW_COCOA_CHDIR_RESOURCES, GLFW_FALSE);

    if (!glfwInit()) {
        std::cerr << "ERROR::GLFW Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Set the required OpenGL version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(800, 600, "3D OBJ Loader", NULL, NULL);
    if (!window) {
        std::cerr << "ERROR::GLFW Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE; 
    if (glewInit() != GLEW_OK) {
        std::cerr << "ERROR::GLEW Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Set up camera and callbacks
    Camera camera(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
    glfwSetWindowUserPointer(window, &camera);
    glfwSetCursorPosCallback(window, mouseCallback);

    // Compile shaders and create shader program
    GLuint shaderProgram = compileShaders();

    // Set up projection matrix
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Process input
        currentDeltaTime = 0.016f;
        camera.processKeyboard(0.016f); // Adjust deltaTime as needed

        // Clear the buffers
        ClearColor();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use the shader program
        glUseProgram(shaderProgram);

        // Set the view and projection matrices
        glm::mat4 view = camera.getViewMatrix();
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Render all loaded models
        renderModels(shaderProgram);
        DoAllTweenRotate();
        DoAllTweenMove();

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

extern "C" DLLEXPORT std::string helper createWin(const std::vector<std::string>& args) {
    createNewWindow();

    return "Window created";
}

extern "C" open DLLEXPORT std::vector<std::string> helper listFunctions() {
    return {std::string("createWin")};
}

extern "C" DLLEXPORT FunctionPtr helper getFunction(const char* name) {
    if (std::string(name) == "createWin") {
        return &createWin;
    }
    return nullptr;
}