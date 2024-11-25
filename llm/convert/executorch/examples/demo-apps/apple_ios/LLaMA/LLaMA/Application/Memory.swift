//
//  Memory.swift
//  LLaMA
//
//  Created by Tom Genlis on 25/11/2024.
//

// Memory management for prompt history
import Foundation

class MemoryManager: ObservableObject {
  @Published var memory: [String: String] = [:]

  func addToMemory(prompt: String, response: String) {
    memory[prompt] = response
  }

  func getMemory() -> [String: String] {
    return memory
  }
}
