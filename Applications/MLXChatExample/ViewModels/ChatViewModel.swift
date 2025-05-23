//
//  ChatViewModel.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import Foundation
import MLXLMCommon
import UniformTypeIdentifiers

/// ViewModel that manages the chat interface and coordinates with MLXService for text generation.
/// Handles user input, message history, media attachments, and generation state.
@Observable
@MainActor
class ChatViewModel {
    /// Service responsible for ML model operations
    private let mlxService: MLXService

    var selectedModel: LMModel

    init(mlxService: MLXService) {
        self.mlxService = mlxService
        guard let model = MLXService.availableModels.first else {
            fatalError("❌ No available models found in MLXService.availableModels.")
        }
        self.selectedModel = model
    }

    /// Current user input text
    var prompt: String = ""

    /// Chat history containing system, user, and assistant messages
    var messages: [Message] = [
        .system("You are Pranam, a medical assistant created by Indian Health-Tech company Apna Vaidya to guide users on their healthcare journey and to answer questions regarding their healthcare concerns. If asked a non-medical question, politely explain that you can only answer health-related questions.When asked for advice, ask for user details like name, age, gender, and symptoms.If the user is not willing to provide details, politely explain that you need the details to provide accurate advice.Tailor all responses to the user's age, gender, and health history and to an Indian context.Do not be too verbose. Make responses concise but informative.")
    ]


    /// Manages image and video attachments for the current message
    var mediaSelection = MediaSelection()

    /// Indicates if text generation is in progress
    var isGenerating = false

    /// Current generation task, used for cancellation
    private var generateTask: Task<Void, any Error>?

    /// Stores performance metrics from the current generation
    private var generateCompletionInfo: GenerateCompletionInfo?

    /// Current generation speed in tokens per second
    var tokensPerSecond: Double {
        generateCompletionInfo?.tokensPerSecond ?? 0
    }

    /// Progress of the current model download, if any
    var modelDownloadProgress: Progress? {
        mlxService.modelDownloadProgress
    }

    /// Most recent error message, if any
    var errorMessage: String?

    /// Indicates if the model is ready for generation
    var isModelReady = false

    /// Indicates if the model is currently being prepared
    var isPreparingModel = false

    /// Generates response for the current prompt and media attachments
    func generate() async {
        // Cancel any existing generation task
        if let existingTask = generateTask {
            existingTask.cancel()
            generateTask = nil
        }

        isGenerating = true

        // Add user message with any media attachments
        messages.append(.user(prompt, images: mediaSelection.images, videos: mediaSelection.videos))
        // Add empty assistant message that will be filled during generation
        messages.append(.assistant(""))

        // Clear the input after sending
        clear(.prompt)

        generateTask = Task {
            // Process generation chunks and update UI
            for await generation in try await mlxService.generate(
                messages: messages, model: selectedModel)
            {
                switch generation {
                case .chunk(let chunk):
                    // Append new text to the current assistant message
                    if let assistantMessage = messages.last {
                        assistantMessage.content += chunk
                    }
                case .info(let info):
                    // Update performance metrics
                    generateCompletionInfo = info
                    isModelReady = true
                }
            }
        }

        do {
            // Handle task completion and cancellation
            try await withTaskCancellationHandler {
                try await generateTask?.value
            } onCancel: {
                Task { @MainActor in
                    generateTask?.cancel()

                    // Mark message as cancelled
                    if let assistantMessage = messages.last {
                        assistantMessage.content += "\n[Cancelled]"
                    }
                }
            }
        } catch {
            errorMessage = error.localizedDescription
            isModelReady = false
        }

        isGenerating = false
        generateTask = nil
    }

    /// Processes and adds media attachments to the current message
    func addMedia(_ result: Result<URL, any Error>) {
        do {
            let url = try result.get()

            // Determine media type and add to appropriate collection
            if let mediaType = UTType(filenameExtension: url.pathExtension) {
                if mediaType.conforms(to: .image) {
                    mediaSelection.images = [url]
                } else if mediaType.conforms(to: .movie) {
                    mediaSelection.videos = [url]
                }
            }
        } catch {
            errorMessage = "Failed to load media item.\n\nError: \(error)"
        }
    }

    /// Clears various aspects of the chat state based on provided options
    func clear(_ options: ClearOption) {
        if options.contains(.prompt) {
            prompt = ""
            mediaSelection = .init()
        }

        if options.contains(.chat) {
            messages = []
            generateTask?.cancel()
            isModelReady = false
        }

        if options.contains(.meta) {
            generateCompletionInfo = nil
        }

        errorMessage = nil
    }

    /// Preloads the model from the hub and marks it as ready or sets error
    func prepareModel() async {
        isPreparingModel = true
        defer { isPreparingModel = false }
        do {
            _ = try await mlxService.generate(messages: messages, model: selectedModel)
            isModelReady = true
        } catch {
            errorMessage = error.localizedDescription
            isModelReady = false
        }
    }
}

/// Manages the state of media attachments in the chat
@Observable
class MediaSelection {
    /// Controls visibility of media selection UI
    var isShowing = false

    /// Currently selected image URLs
    var images: [URL] = []

    /// Currently selected video URLs
    var videos: [URL] = []

    /// Whether any media is currently selected
    var isEmpty: Bool {
        images.isEmpty && videos.isEmpty
    }
}

/// Options for clearing different aspects of the chat state
struct ClearOption: RawRepresentable, OptionSet {
    let rawValue: Int

    /// Clears current prompt and media selection
    static let prompt = ClearOption(rawValue: 1 << 0)
    /// Clears chat history and cancels generation
    static let chat = ClearOption(rawValue: 1 << 1)
    /// Clears generation metadata
    static let meta = ClearOption(rawValue: 1 << 2)
}

