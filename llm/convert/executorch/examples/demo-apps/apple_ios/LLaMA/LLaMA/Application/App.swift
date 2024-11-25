/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import SwiftUI

@main
struct App: SwiftUI.App {
  var body: some Scene {
    WindowGroup {
      ContentView()
        .onAppear {
            print("Loading CSV file...")
            if let filePath = Bundle.main.path(forResource: "dataset", ofType: "csv") {
                let csvData = readCSV(filePath: filePath)
                
                let inputString = "In Lovely Blueness (number one) and Nu sur fond rouge are in this string"
                print(inputString)
                let ragInput = createRAGInput(forString: inputString, csvEntries: csvData)
                print(ragInput)
            } else {
                print("CSV file not found in the bundle.")
            }
        }
    }
  }
}
