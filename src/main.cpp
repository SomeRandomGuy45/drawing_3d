#include "main_lib.h"
#include <atomic>
#include <condition_variable>
#include <mutex>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::atomic<bool> windowReady(false);
std::condition_variable cv;
std::mutex mtx;

GLFWwindow* window;

bool locked = true;

std::vector<std::tuple<GLuint, std::pair<unsigned int, glm::vec3>, std::tuple<glm::vec3, glm::vec3, glm::vec3>, GLuint>> models;

// Camera class
class Camera {
public:
    Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
        : Position(position), WorldUp(up), Yaw(yaw), Pitch(pitch) {
        updateCameraVectors();
    }

    void setProjection(float aspectRatio) {
        Projection = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 100.0f);
    }

    glm::mat4 getViewMatrix() const {
        return glm::lookAt(Position, Position + Front, Up);
    }

    glm::mat4 getProjectionMatrix() const {
        return Projection;
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

    glm::mat4 Projection;

    float Yaw;
    float Pitch;
};

GLuint loadTexture(const std::string& path) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    
    // Load image
    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true); // Flip loaded texture coordinates
    unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
    
    if (data) {
        GLenum format = (nrChannels == 1) ? GL_RED : (nrChannels == 3) ? GL_RGB : GL_RGBA;
        glBindTexture(GL_TEXTURE_2D, textureID);
        
        // Generate texture
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    } else {
        std::cerr << "ERROR::IMAGE-LOADING Failed to load texture: " << path << std::endl;
    }

    stbi_image_free(data); // Free the image data
    return textureID;
}

// Function to load an OBJ file using Assimp
std::tuple<GLuint, std::pair<unsigned int, glm::vec3>, std::tuple<glm::vec3, glm::vec3, glm::vec3>, GLuint> loadModel(
    const std::string& path, 
    glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), 
    glm::vec3 color = glm::vec3(0.0f, 0.0f, 0.0f), 
    glm::vec3 scale = glm::vec3(1.0f), 
    glm::vec3 rotationAxis = glm::vec3(0.0f, 0.0f, 0.0f),
    const std::string& texturePath = ""
)
{
    GLuint textureID = 0;
    std::vector<GLuint> indices;
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> texCoords;

    color.x /= 255.0f;
    color.y /= 255.0f;
    color.z /= 255.0f;

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_FixInfacingNormals | aiProcess_SortByPType);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
        return { 0, { 0, glm::vec3(0.0f) }, { glm::vec3(1.0f), glm::vec3(1.0f), glm::vec3(0.0f) }, 0 }; 
    }

    // Process each mesh in the scene
    for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[i];

        // Process vertices and texture coordinates
        for (unsigned int j = 0; j < mesh->mNumVertices; j++) {
            aiVector3D pos = mesh->mVertices[j];
            vertices.emplace_back(pos.x, pos.y, pos.z);

            if (mesh->mTextureCoords[0]) {
                aiVector3D texCoord = mesh->mTextureCoords[0][j];
                texCoords.emplace_back(texCoord.x, texCoord.y);
            } else {
                texCoords.emplace_back(0.0f, 0.0f);
            }
        }

        // Process indices
        for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
            aiFace face = mesh->mFaces[j];
            for (unsigned int k = 0; k < face.mNumIndices; k++) {
                indices.push_back(face.mIndices[k]);
            }
        }

        // Load material properties
        if (mesh->mMaterialIndex >= 0) {
            aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
            aiColor3D diffuse;
            material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
            color = glm::vec3(diffuse.r, diffuse.g, diffuse.b); // Use the diffuse color

            // Load the texture if available
            aiString texturePath;
            if (material->GetTexture(aiTextureType_DIFFUSE, 0, &texturePath) == AI_SUCCESS) {
                // Load texture
                std::string fullPath = std::string(texturePath.C_Str());
                std::cout << "INFO::IMAGE Loading Image Path:" << fullPath << "\n";
                textureID = loadTexture(fullPath);
                // Save the textureID or use it directly if needed
            }
        }
    }

    std::cout << "INFO::IMAGE Loaded " << vertices.size() << " vertices and " << indices.size() << " indices." << std::endl;

    GLuint VAO, VBO, EBO, TBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &TBO);

    glBindVertexArray(VAO);

    // Vertex Buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    // Texture Coordinate Buffer
    glBindBuffer(GL_ARRAY_BUFFER, TBO);
    glBufferData(GL_ARRAY_BUFFER, texCoords.size() * sizeof(glm::vec2), texCoords.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (GLvoid*)0);
    glEnableVertexAttribArray(1);

    // Element Buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    if (!texturePath.empty() && textureID == 0) {
        textureID = loadTexture(texturePath);
    }

    return { VAO, { static_cast<unsigned int>(indices.size()), position }, { color, scale, rotationAxis }, textureID };
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
        GLuint textureID = std::get<3>(model); // Get textureID

        // Normalize the axis and calculate the angle
        float rotationAngle = glm::length(rotationAxis);
        if (rotationAngle > 0.0f) {
            rotationAxis = glm::normalize(rotationAxis);
        }

        glm::mat4 modelMatrix = glm::translate(glm::mat4(1.0f), position);
        modelMatrix = glm::rotate(modelMatrix, rotationAngle, rotationAxis); // Rotation
        modelMatrix = glm::scale(modelMatrix, scale); // Scaling
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(modelMatrix));

        // Check if the texture is used
        glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), textureID != 0); // Set the useTexture uniform

        // Set color for the fragment shader
        glUniform3fv(glGetUniformLocation(shaderProgram, "color"), 1, glm::value_ptr(color));

        // Bind the texture if textureID is not zero
        if (textureID != 0) {
            glActiveTexture(GL_TEXTURE0); // Activate texture unit
            glBindTexture(GL_TEXTURE_2D, textureID); // Bind texture
        }

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // Unbind the texture
        if (textureID != 0) {
            glBindTexture(GL_TEXTURE_2D, 0);
        }
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
layout(location = 1) in vec2 texCoords; // Add texture coordinates input

out vec2 fragTexCoords; // Pass texture coordinates to fragment shader

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    fragTexCoords = texCoords; // Pass texture coordinates to fragment shader
}
)";

// Fragment Shader
const char* fragmentShaderSource = R"(
#version 330 core
in vec2 fragTexCoords; // Receive texture coordinates from vertex shader
out vec4 outColor;

uniform sampler2D texture1; // Texture sampler
uniform vec3 color; // Color uniform
uniform bool useTexture; // Boolean to determine whether to use texture or color

void main() {
    if (useTexture) {
        vec4 textureColor = texture(texture1, fragTexCoords); // Sample the texture
        outColor = textureColor; // Use texture color
    } else {
        outColor = vec4(color, 1.0); // Use the specified color
    }
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

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    // Adjust the viewport based on the new width and height
    glViewport(0, 0, width, height);

    // Get camera from user pointer
    Camera* camera = static_cast<Camera*>(glfwGetWindowUserPointer(window));
    
    // Update the projection matrix
    camera->setProjection((float)width / (float)height);
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
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "ERROR::GLFW Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Set the required OpenGL version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(800, 600, "Main App", NULL, NULL);
    if (!window) {
        std::cerr << "ERROR::GLFW Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Check OpenGL version
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version = glGetString(GL_VERSION);
    std::cout << "INFO::GLFW Renderer: " << renderer << std::endl;
    std::cout << "INFO::GLFW OpenGL version supported: " << version << std::endl;

    // Initialize GLEW
    glewExperimental = GL_TRUE; 
    if (glewInit() != GLEW_OK) {
        std::cerr << "ERROR::GLEW Failed to initialize GLEW" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST); // Enable depth testing

    // Set up camera
    Camera camera(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
    glfwSetWindowUserPointer(window, &camera);
    glfwSetCursorPosCallback(window, mouseCallback);

    // Set the framebuffer size callback
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Compile shaders and create shader program
    GLuint shaderProgram = compileShaders();

    // Load models into a vector
    //Example: models.push_back(loadModel("pathToObj.obj", glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(255.0f,0.0f,0.0f), glm::vec3(1.0f), glm::vec3(90.0f, 45.0f, 90.0f)));

    // Set up projection matrix
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

    glEnable(GL_DEPTH_TEST); // Enable depth testing
    glEnable(GL_CULL_FACE);  // Enable backface culling
    glCullFace(GL_BACK);      // Cull back faces

    // Main loop
    float lastFrameTime = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        // Process input
        float currentTime = glfwGetTime();
        currentDeltaTime = currentTime - lastFrameTime; // Calculate delta time
        lastFrameTime = currentTime; // Update last frame time
        camera.processKeyboard(currentDeltaTime); // Adjust deltaTime as needed

        // Clear the buffers
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use the shader program
        glUseProgram(shaderProgram);

        // Set the view and projection matrices
        glm::mat4 view = camera.getViewMatrix();
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        
        glm::mat4 projection = camera.getProjectionMatrix();
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
    for (const auto& model : models) {
        GLuint VAO = std::get<0>(model); // Extract the VAO from the model tuple
        glDeleteVertexArrays(1, &VAO);
    }
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

extern "C" DLLEXPORT std::string helper createWin(const std::vector<std::string>& args) {
    std::thread t = std::thread([&]() {
        createNewWindow();
    })
    t.detach();

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
