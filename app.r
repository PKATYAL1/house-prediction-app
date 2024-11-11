library(shiny)
library(reticulate)

use_virtualenv("r-reticulate", required = TRUE)

py_install("joblib")
py_install("xgboost")
py_install("pandas")

# Load the original model
xgb_model <- import("joblib")$load("xgboost_model.joblib")
pd <- import("pandas")

# Helper function to make predictions with the original model
predict_house_price <- function(living_area, bathrooms, bedrooms, latitude, longitude) {
  user_input <- data.frame(
    Latitude = as.numeric(latitude),
    Longitude = as.numeric(longitude),
    `Living Area` = as.numeric(living_area),
    Bathrooms = as.numeric(bathrooms),
    Bedrooms = as.numeric(bedrooms)
  )
  
  colnames(user_input) <- c("Latitude", "Longitude", "Living Area", "Bathrooms", "Bedrooms")
  
  py_user_input <- pd$DataFrame(user_input)
  

  prediction <- xgb_model$predict(py_user_input)
  return(prediction[1])
}

# Define the UI
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body { font-family: Arial, sans-serif; }
      .btn-primary { background-color: #007BFF; color: white; }
      .container { max-width: 700px; }
      .prediction-box { background-color: #f8f9fa; padding: 20px; border-radius: 5px; border: 1px solid #ddd; }
    "))
  ),
  
  titlePanel("House Price Prediction Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      sliderInput("living_area", "Living Area (sqft)", min = 1000, max = 5000, value = 1000, step = 100),
      selectInput("bathrooms", "Bathrooms", choices = 1:10, selected = 2),
      selectInput("bedrooms", "Bedrooms", choices = 1:10, selected = 3),
      checkboxInput("advanced_options", "Show Advanced Options", value = FALSE),
      
      # Show latitude and longitude inputs conditionally
      conditionalPanel(
        condition = "input.advanced_options == true",
        numericInput("latitude", "Latitude", value = 37.7749),
        numericInput("longitude", "Longitude", value = -122.4194)
      ),
      
      actionButton("predict", "Predict", class = "btn-primary")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Plain Prediction",
                 div(class = "prediction-box",
                     h3("Prediction Result"),
                     verbatimTextOutput("prediction")
                 )
        ),
        tabPanel("Line Graph",
                 plotOutput("trendPlot")  # Reactive line graph
        ),
        tabPanel("Bar Graph",
                 plotOutput("bedroomPricePlot")  # Reactive bar chart
        ),
        tabPanel("Comparison",
                 h3("Comparison with AVM"),
                 tableOutput("comparisonTable")
        ),
        tabPanel("Instructions",
                 helpText("Use the sidebar to adjust Living Area, Bathrooms, and Bedrooms, then explore predictions in different views.")
        ),
        tabPanel("About",
                 helpText("This dashboard provides dynamic house price predictions based on input values for Living Area, Longitude, Latitude, Bathrooms, and Bedrooms.")
        )
      )
    )
  )
)

# Define server logic
server <- function(input, output) {
  

  observeEvent(input$predict, {
    prediction <- predict_house_price(input$living_area, input$bathrooms, input$bedrooms, 
                                      ifelse(input$advanced_options, input$latitude, 37.7749), 
                                      ifelse(input$advanced_options, input$longitude, -122.4194))
    
    output$prediction <- renderText({
      paste("Predicted House Price: $", scales::dollar(prediction))
    })
    

    output$comparisonTable <- renderTable({
      data.frame(
        Model = c("Main Model Prediction", "AVM Prediction"),
        Predicted_Price = c(scales::dollar(prediction), "$500,000")  # Placeholder value for AVM
      )
    })
  })
  
  #  price predictions across different Living Areas for line graph
  reactive_line_predictions <- reactive({
    living_area_values <- seq(500, 5000, by = 100)
    predictions <- sapply(living_area_values, function(living_area) {
      predict_house_price(living_area, input$bathrooms, input$bedrooms, 
                          ifelse(input$advanced_options, input$latitude, 37.7749), 
                          ifelse(input$advanced_options, input$longitude, -122.4194))
    })
    data.frame(LivingArea = living_area_values, PredictedPrice = predictions)
  })
  
  #  line plot to show trend in predicted prices based on Living Area
  output$trendPlot <- renderPlot({
    trend_data <- reactive_line_predictions()
    plot(trend_data$LivingArea, trend_data$PredictedPrice, type = "l", col = "blue", lwd = 2,
         main = "Predicted Price Trend based on Living Area",
         xlab = "Living Area (sqft)", ylab = "Predicted Price ($)")
  })
  
  # Predict price for each number of bedrooms
  reactive_bedroom_prices <- reactive({
    bedrooms_range <- 1:10
    predictions <- sapply(bedrooms_range, function(bedrooms) {
      predict_house_price(input$living_area, input$bathrooms, bedrooms, 
                          ifelse(input$advanced_options, input$latitude, 37.7749), 
                          ifelse(input$advanced_options, input$longitude, -122.4194))
    })
    data.frame(Bedrooms = bedrooms_range, PredictedPrice = predictions)
  })
  
  #  bar plot to show predicted prices across bedroom counts
  output$bedroomPricePlot <- renderPlot({
    bedroom_data <- reactive_bedroom_prices()
    barplot(bedroom_data$PredictedPrice, names.arg = bedroom_data$Bedrooms,
            col = "skyblue", border = "white", 
            main = "Predicted Price by Number of Bedrooms",
            xlab = "Number of Bedrooms", ylab = "Predicted Price ($)")
  })
}

# Run the application 
shinyApp(ui = ui, server = server)


