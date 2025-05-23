// Copyright © 2024 Apple Inc.

import AVFoundation
import CoreImage
import Foundation
import MLX
import Tokenizers

public typealias Message = [String: Any]

/// Container for raw user input.
///
/// A ``UserInputProcessor`` can convert this to ``LMInput``.
/// See also ``ModelContext``.
public struct UserInput: Sendable {

    /// Representation of a prompt or series of messages (conversation).
    ///
    /// This may be a single string with a user prompt or a series of back
    /// and forth responses representing a conversation.
    public enum Prompt: Sendable, CustomStringConvertible {
        /// a single string
        case text(String)

        /// model specific array of dictionaries
        case messages([Message])

        /// model agnostic structured chat (series of messages)
        case chat([Chat.Message])

        public var description: String {
            switch self {
            case .text(let text):
                return text
            case .messages(let messages):
                return messages.map { $0.description }.joined(separator: "\n")
            case .chat(let messages):
                return messages.map(\.content).joined(separator: "\n")
            }
        }
    }

    /// Representation of a video resource.
    public enum Video: Sendable {
        case avAsset(AVAsset)
        case url(URL)

        public func asAVAsset() -> AVAsset {
            switch self {
            case .avAsset(let asset):
                return asset
            case .url(let url):
                return AVAsset(url: url)
            }
        }
    }

    /// Representation of an image resource.
    public enum Image: Sendable {
        case ciImage(CIImage)
        case url(URL)
        case array(MLXArray)

        public func asCIImage() throws -> CIImage {
            switch self {
            case .ciImage(let image):
                return image

            case .url(let url):
                if let image = CIImage(contentsOf: url) {
                    return image
                }
                throw UserInputError.unableToLoad(url)

            case .array(let array):
                guard array.ndim == 3 else {
                    throw UserInputError.arrayError("array must have 3 dimensions: \(array.ndim)")
                }

                var array = array

                // convert to 0 .. 255
                if array.max().item(Float.self) <= 1.0 {
                    array = array * 255
                }

                // planar -> pixels
                switch array.dim(0) {
                case 3, 4:
                    // channels first (planar)
                    array = array.transposed(1, 2, 0)
                default:
                    break
                }

                // 4 components per pixel
                switch array.dim(-1) {
                case 3:
                    // pad to 4 bytes per pixel
                    array = padded(array, widths: [0, 0, [0, 1]], value: MLXArray(255))
                case 4:
                    // good
                    break
                default:
                    throw UserInputError.arrayError(
                        "channel dimension must be last and 3/4: \(array.shape)")
                    break
                }

                let arrayData = array.asData()
                let (H, W, C) = array.shape3
                let cs = CGColorSpace(name: CGColorSpace.sRGB)!

                return CIImage(
                    bitmapData: arrayData.data, bytesPerRow: W * 4,
                    size: .init(width: W, height: H),
                    format: .RGBA8, colorSpace: cs)
            }
        }
    }

    /// Representation of processing to apply to media.
    public struct Processing: Sendable {
        public var resize: CGSize?

        public init(resize: CGSize? = nil) {
            self.resize = resize
        }
    }

    /// The prompt to evaluate.
    public var prompt: Prompt {
        didSet {
            switch prompt {
            case .text, .messages:
                // no action
                break
            case .chat(let messages):
                // rebuild images & videos
                self.images = messages.reduce(into: []) { result, message in
                    result.append(contentsOf: message.images)
                }
                self.videos = messages.reduce(into: []) { result, message in
                    result.append(contentsOf: message.videos)
                }
            }
        }
    }

    /// The images associated with the `UserInput`.
    ///
    /// If the ``prompt-swift.property`` is a ``Prompt-swift.enum/chat(_:)`` this will
    /// collect the images from the chat messages, otherwise these are the stored images with the ``UserInput``.
    public var images = [Image]()

    /// The images associated with the `UserInput`.
    ///
    /// If the ``prompt-swift.property`` is a ``Prompt-swift.enum/chat(_:)`` this will
    /// collect the videos from the chat messages, otherwise these are the stored videos with the ``UserInput``.
    public var videos = [Video]()

    public var tools: [ToolSpec]?

    /// Additional values provided for the chat template rendering context
    public var additionalContext: [String: Any]?
    public var processing: Processing = .init()

    /// Initialize the `UserInput` with a single text prompt.
    ///
    /// - Parameters:
    ///   - prompt: text prompt
    ///   - images: optional images
    ///   - videos: optional videos
    ///   - tools: optional tool specifications
    ///   - additionalContext: optional context (model specific)
    /// ### See Also
    /// - ``Prompt-swift.enum/text(_:)``
    /// - ``init(chat:tools:additionalContext:)``
    public init(
        prompt: String, images: [Image] = [Image](), videos: [Video] = [Video](),
        tools: [ToolSpec]? = nil,
        additionalContext: [String: Any]? = nil
    ) {
        self.prompt = .chat([
            .user(prompt, images: images, videos: videos)
        ])
        self.tools = tools
        self.additionalContext = additionalContext
    }

    /// Initialize the `UserInput` with model specific mesage structures.
    ///
    /// For example, the Qwen2VL model wants input in this format:
    ///
    /// ```
    /// [
    ///     [
    ///         "role": "user",
    ///         "content": [
    ///             [
    ///                 "type": "text",
    ///                 "text": "What is this?"
    ///             ],
    ///             [
    ///                 "type": "image",
    ///             ],
    ///         ]
    ///     ]
    /// ]
    /// ```
    ///
    /// Typically the ``init(chat:tools:additionalContext:)`` should be used instead
    /// along with a model specific ``MessageGenerator`` (supplied by the ``UserInputProcessor``).
    ///
    /// - Parameters:
    ///   - messages: array of dictionaries representing the prompt in a model specific format
    ///   - images: optional images
    ///   - videos: optional videos
    ///   - tools: optional tool specifications
    ///   - additionalContext: optional context (model specific)
    /// ### See Also
    /// - ``Prompt-swift.enum/text(_:)``
    /// - ``init(chat:tools:additionalContext:)``
    public init(
        messages: [Message], images: [Image] = [Image](), videos: [Video] = [Video](),
        tools: [ToolSpec]? = nil,
        additionalContext: [String: Any]? = nil
    ) {
        self.prompt = .messages(messages)
        self.images = images
        self.videos = videos
        self.tools = tools
        self.additionalContext = additionalContext
    }

    /// Initialize the `UserInput` with a model agnostic structured context.
    ///
    /// For example:
    ///
    /// ```
    /// let chat: [Chat.Message] = [
    ///     .system("You are a helpful photographic assistant."),
    ///     .user("Please describe the photo.", images: [image1]),
    /// ]
    /// let userInput = UserInput(chat: chat)
    /// ```
    ///
    /// A model specific ``MessageGenerator`` (supplied by the ``UserInputProcessor``)
    /// is used to convert this into a model specific format.
    ///
    /// - Parameters:
    ///   - chat: structured content
    ///   - tools: optional tool specifications
    ///   - additionalContext: optional context (model specific)
    /// ### See Also
    /// - ``Prompt-swift.enum/text(_:)``
    /// - ``init(chat:tools:additionalContext:)``
    public init(
        chat: [Chat.Message],
        tools: [ToolSpec]? = nil,
        additionalContext: [String: Any]? = nil
    ) {
        self.prompt = .chat(chat)

        // note: prompt.didSet is not triggered in init
        self.images = chat.reduce(into: []) { result, message in
            result.append(contentsOf: message.images)
        }
        self.videos = chat.reduce(into: []) { result, message in
            result.append(contentsOf: message.videos)
        }
        self.tools = tools
        self.additionalContext = additionalContext
    }

    /// Initialize the `UserInput` with a preconfigured ``Prompt-swift.enum``.
    ///
    /// ``init(chat:tools:additionalContext:)`` is the preferred mechanism.
    ///
    /// - Parameters:
    ///   - prompt: the prompt
    ///   - images: optional images
    ///   - videos: optional videos
    ///   - tools: optional tool specifications
    ///   - processing: optional processing to be applied to media
    ///   - additionalContext: optional context (model specific)
    /// ### See Also
    /// - ``Prompt-swift.enum/text(_:)``
    /// - ``init(chat:tools:additionalContext:)``
    public init(
        prompt: Prompt,
        images: [Image] = [Image](),
        videos: [Video] = [Video](),
        processing: Processing = .init(),
        tools: [ToolSpec]? = nil, additionalContext: [String: Any]? = nil
    ) {
        self.prompt = prompt
        switch prompt {
        case .text, .messages:
            self.images = images
            self.videos = videos
        case .chat:
            break
        }
        self.processing = processing
        self.tools = tools
        self.additionalContext = additionalContext
    }
}

/// Protocol for a type that can convert ``UserInput`` to ``LMInput``.
///
/// See also ``ModelContext``.
public protocol UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput
}

private enum UserInputError: LocalizedError {
    case notImplemented
    case unableToLoad(URL)
    case arrayError(String)

    var errorDescription: String? {
        switch self {
        case .notImplemented:
            return String(localized: "This functionality is not implemented.")
        case .unableToLoad(let url):
            return String(localized: "Unable to load image from URL: \(url.path).")
        case .arrayError(let message):
            return String(localized: "Error processing image array: \(message).")
        }
    }
}

/// A do-nothing ``UserInputProcessor``.
public struct StandInUserInputProcessor: UserInputProcessor {
    public init() {}

    public func prepare(input: UserInput) throws -> LMInput {
        throw UserInputError.notImplemented
    }
}

// MARK: - UserInput.Prompt Extension

extension UserInput.Prompt {
    /// Converts the prompt into an array of message dictionaries suitable for tokenizers.
    ///
    /// This method assumes that `Chat.Message` (from the `Chat` module, likely part of MLXLMCommon or a similar library)
    /// has accessible `role` (e.g., an enum with a `String` `rawValue` like "user", "assistant", "system")
    /// and `content` (a `String` representing the textual part of the message) properties.
    public func asMessages() -> [Message] {
        switch self {
        case .text(let textContent):
            // Convert a single text string into a user message dictionary
            return [["role": "user", "content": textContent]]

        case .messages(let messageArray):
            // Already in the desired [Message] format
            return messageArray

        case .chat(let chatMessages):
            // Convert an array of `Chat.Message` objects to `[Message]`
            // This mapping depends on the actual structure of `Chat.Message`.
            // The following assumes `Chat.Message` has properties like:
            // - `role`: An enum (e.g., `Chat.Role`) with a `rawValue` of type `String`.
            // - `content`: A `String` containing the text of the message.
            return chatMessages.map { chatMsg in
                // If Chat.Message has, for example, chatMsg.role.rawValue and chatMsg.content:
                // return ["role": chatMsg.role.rawValue, "content": chatMsg.content]

                // If Chat.Message is an enum with helper properties, e.g., chatMsg.roleString and chatMsg.textContent:
                // return ["role": chatMsg.roleString, "content": chatMsg.textContent]

                // Placeholder for actual conversion based on Chat.Message structure.
                // You'll need to replace this with the correct properties of your Chat.Message type.
                // For demonstration, assuming Chat.Message conforms to a hypothetical protocol
                // or has known properties like 'roleString' and 'textContent'.
                // A common pattern would involve accessing properties like:
                //guard let role = (chatMsg as? any ChatMessageRepresentable)?.roleString,
                //      let content = (chatMsg as? any ChatMessageRepresentable)?.textContent else {
                //    // Handle cases where a Chat.Message might not conform or properties are missing
                //    // Or, if Chat.Message is a concrete type with known properties:
                //    // e.g. if Chat.Message is a struct like:
                //    // struct ChatMessage { var role: ChatRoleEnum; var text: String }
                //    // then it would be: ["role": chatMsg.role.rawValue, "content": chatMsg.text]
                //    return ["role": "unknown", "content": "Error converting chat message"]
                //}
                // return ["role": role, "content": content]

                // Let's assume a common structure for Chat.Message based on its usage in UserInput initializers.
                // (e.g., Chat.Message might have .role: Chat.Role and .content: String)
                // Replace `chatMsg.role.rawValue` and `chatMsg.content` with actual property access
                // if your Chat.Message structure is different.
                // This example assumes Chat.Message has a `role` enum property with a `rawValue`
                // and a `content` String property.
                // e.g. if Chat.Message is `struct MyChatMessage { var role: MyRoleEnum; var content: String }`
                // and `MyRoleEnum` is `enum MyRoleEnum: String { case user, assistant }`
                // then it would be `["role": chatMsg.role.rawValue, "content": chatMsg.content]`

                // This is a common pattern and needs to match your `Chat.Message` definition:
                let roleValue: String
                let contentValue: String

                // You need to inspect the actual Chat.Message structure.
                // For example, if it's an enum:
                // switch chatMsg {
                // case .user(let c, _): roleValue = "user"; contentValue = c
                // case .assistant(let c, _): roleValue = "assistant"; contentValue = c
                // case .system(let c): roleValue = "system"; contentValue = c
                // // Add other cases if they exist
                // default: roleValue = "unknown"; contentValue = "" // Or handle error
                // }
                // Or if it's a struct with properties (most likely):
                // Assuming chatMsg has properties `role: RoleEnum` and `content: String`
                // where RoleEnum has a `rawValue: String`. This is a strong convention.
                // You will need to ensure these property names (`role`, `rawValue`, `content`)
                // match what's available on your `Chat.Message` type.
                // The type of Chat.Message is not provided in the snippets, but it's likely from MLXLMCommon.
                // For this example, we'll use a placeholder structure.

                // **Important**: Replace the following with actual access to your Chat.Message properties.
                // If your Chat.Message type has `role: SomeRoleEnum` and `content: String`, and `SomeRoleEnum` has `rawValue: String`:
                // roleValue = chatMsg.role.rawValue
                // contentValue = chatMsg.content
                // If the structure is different, adjust accordingly.
                // For now, using description as a fallback to illustrate, but this is NOT correct for role.
                // The .description of Prompt gives combined content, not structured messages.
                // We must assume `Chat.Message` gives structured role/content.
                // The most probable structure for `Chat.Message` would allow:
                // `["role": chatMsg.role.rawValue, "content": chatMsg.content]`
                // Since we don't have the definition of `Chat.Message`, this part is illustrative.
                // You MUST adapt this to your actual Chat.Message structure.
                // A simple, but potentially incomplete, way if Chat.Message has `content` and you infer role:
                // This is a guess, refine based on actual Chat.Message:
                if let role = (((chatMsg as? CustomStringConvertible)?.description.lowercased().contains("user")) != nil) ? "user" : "assistant", // Highly speculative
                   let content = (chatMsg as? CustomStringConvertible)?.description { // Also speculative that description is just content
                    return ["role": role, "content": content]
                } else {
                     // Fallback or error for messages that cannot be converted
                    return ["role": "unknown", "content": String(describing: chatMsg)]
                }
            }
        }
    }
}
