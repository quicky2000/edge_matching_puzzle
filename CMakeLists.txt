#-----------------------------------------
# CMakeLists Skeleton 2020/12/10
#-----------------------------------------
cmake_minimum_required(VERSION 3.11.4)
project(edge_matching_puzzle LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 17)
set(ENABLE_CUDA_CODE off)

set(MY_SOURCE_FILES
    include/algo_based_feature.h
    include/border_backtracker.h
    include/border_color_constraint.h
    include/border_constraint_generator.h
    include/border_exploration.h
    include/CUDA_backtracker_stack.h
    include/CUDA_common.h
    include/CUDA_glutton_max_stack.h
    include/CUDA_memory_managed_array.h
    include/CUDA_memory_managed_pointer.h
    include/CUDA_piece_position_info.h
    include/CUDA_piece_position_info2.h
    include/CUDA_piece_position_info_base.h
    include/CUDA_situation_capability.h
    include/CUDA_transition_manager.h
    include/emp_advanced_feature_base.h
    include/emp_basic_strategy_generator.h
    include/emp_constraint.h
    include/emp_FSM_context.h
    include/emp_FSM.h
    include/emp_FSM_info.h
    include/emp_FSM_motor.h
    include/emp_FSM_situation_analyzer.h
    include/emp_FSM_situation.h
    include/emp_FSM_transition.h
    include/emp_gui.h
    include/emp_link.h
    include/emp_piece_border.h
    include/emp_piece_constraint.h
    include/emp_piece_corner.h
    include/emp_piece_db.h
    include/emp_piece.h
    include/emp_pieces_parser.h
    include/emp_position_strategy.h
    include/emp_se_step_info.h
    include/emp_simplex_listener.h
    include/emp_situation.h
    include/emp_situation_base.h
    include/emp_situation_binary_dumper.h
    include/emp_situation_binary_reader.h
    include/emp_spiral_strategy_generator.h
    include/emp_strategy_generator_factory.h
    include/emp_strategy_generator.h
    include/emp_strategy.h
    include/emp_stream_strategy_generator.h
    include/emp_text_strategy_generator.h
    include/emp_types.h
    include/emp_variable_generator.h
    include/emp_web_server.h
    include/factory_feature.h
    include/feature_border_exploration.h
    include/feature_compute_stats.h
    include/feature_display_all.h
    include/feature_display_dump.h
    include/feature_display_max.h
    include/feature_display_situation.h
    include/feature_display_solutions.h
    include/feature_dump_solutions.h
    include/feature_dump_summary.h
    include/feature_if.h
    include/feature_profile.h
    include/feature_simplex.h
    include/feature_situation_profile.h
    include/feature_system_equation.h
    include/feature_sys_equa_CUDA_base.h
    include/feature_CUDA_backtracker.h
    include/feature_CUDA_glutton_max.h
    include/light_border_pieces_db.h
    include/octet_array.h
    include/piece_position_info.h
    include/precomputed_constraint.h
    include/precomputed_transition_info.h
    include/sequential_border_backtracker.h
    include/simplex_variable.h
    include/situation_capability.h
    include/situation_profile.h
    include/situation_string_formatter.h
    include/system_equation_for_CUDA.h
    include/transition_manager.h
    include/VTK_histogram_dumper.h
    include/VTK_line_plot_dumper.h
    include/VTK_table_based_dumper.h
    src/border_backtracker.cpp
    src/emp_FSM.cpp
    src/emp_FSM_motor.cpp
    src/emp_FSM_situation_analyzer.cpp
    src/emp_piece.cpp
    src/emp_se_step_info.cpp
    src/emp_situation_base.cpp
    src/emp_strategy_generator_factory.cpp
    src/emp_types.cpp
    src/emp_web_server.cpp
    src/feature_compute_stats.cpp
    src/feature_CUDA_backtracker.cpp
    src/feature_display_all.cpp
    src/feature_display_max.cpp
    src/feature_display_solutions.cpp
    src/feature_dump_solutions.cpp
    src/feature_situation_profile.cpp
    src/feature_system_equation.cpp
   )


if(${ENABLE_CUDA_CODE})
    set (MY_CPP_FLAGS -DENABLE_CUDA_CODE)
    set (MY_SOURCE_FILES ${MY_SOURCE_FILES}
         src/CUDA_backtracker.cu
         src/CUDA_glutton_max.cu
         src/CUDA_piece_position_info_base.cu
        )
else(${ENABLE_CUDA_CODE})
    set (MY_CPP_FLAGS -pedantic)
endif(${ENABLE_CUDA_CODE})

# List header directories in project
set(MY_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}/include
   )

#Create dependancy list
set(DEPENDANCY_LIST "")
LIST(APPEND DEPENDANCY_LIST "quicky_exception")
LIST(APPEND DEPENDANCY_LIST "quicky_utils")
LIST(APPEND DEPENDANCY_LIST "simple_gui")
LIST(APPEND DEPENDANCY_LIST "lib_bmp")
LIST(APPEND DEPENDANCY_LIST "FSM_base")
LIST(APPEND DEPENDANCY_LIST "FSM_framework")
LIST(APPEND DEPENDANCY_LIST "parameter_manager")
LIST(APPEND DEPENDANCY_LIST "EXT_microhttpd")
LIST(APPEND DEPENDANCY_LIST "combinatorics")
LIST(APPEND DEPENDANCY_LIST "simplex")
LIST(APPEND DEPENDANCY_LIST "my_cuda")

#------------------------------
#- Generic part
#------------------------------
# Search for CUDA files
foreach(SOURCE_FILE IN ITEMS ${MY_SOURCE_FILES})
    if(${SOURCE_FILE} MATCHES "(.*\).cu$")
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
        enable_language(CUDA)
        include(CheckLanguage)
        check_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} " --ptxas-options=-v -arch=sm_75 -gencode=arch=compute_75,code=sm_75")
            if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
            else()
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
            endif()
        endif()
    endif()
endforeach(SOURCE_FILE)

set(GENERATED_FILES "")

# Search for Bison files
foreach(SOURCE_FILE IN ITEMS ${MY_SOURCE_FILES})
    if(${SOURCE_FILE} MATCHES "(.*\).ypp$")
        get_filename_component(BISON_NAME ${CMAKE_MATCH_1} NAME)
        message("Bison parser detected: " ${BISON_NAME})
        set(BISON_OUTPUT ${BISON_NAME}.hpp ${BISON_NAME}.cpp)
        string(FIND "$GENERATED_FILES" location.hh GENERIC_BISON_OK)
        if(GENERIC_BISON_OK EQUAL -1)
            set(BISON_OUTPUT ${BISON_OUTPUT} location.hh position.hh stack.hh)
            message("Add generic bison files")
        endif()
        add_custom_command(OUTPUT ${BISON_OUTPUT}
                           COMMAND bison -o ${BISON_NAME}.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
                           DEPENDS ${SOURCE_FILE}
                           COMMENT "Generate Bison parser for \"${SOURCE_FILE}\""
                           )
        set(GENERATED_FILES ${GENERATED_FILES} ${BISON_OUTPUT})
    endif()
endforeach(SOURCE_FILE)

#Search for flex files
foreach(SOURCE_FILE IN ITEMS ${MY_SOURCE_FILES})
    if(${SOURCE_FILE} MATCHES "(.*\).ll$")
        get_filename_component(FLEX_NAME ${CMAKE_MATCH_1} NAME)
        message("Flex lexer detected: " ${FLEX_NAME})
        set(FLEX_OUTPUT ${FLEX_NAME}.h ${FLEX_NAME}.cpp)
        add_custom_command(OUTPUT ${FLEX_OUTPUT}
                           COMMAND flex -o${FLEX_NAME}.cpp --header-file=${FLEX_NAME}.h ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
                           DEPENDS ${SOURCE_FILE}
                           COMMENT "Generate Flex lexer for \"${SOURCE_FILE}\""
                           )
        set(GENERATED_FILES ${GENERATED_FILES} ${FLEX_OUTPUT})
    endif()
endforeach(SOURCE_FILE)

if(GENERATED_FILES)
    set(MY_INCLUDE_DIRECTORIES
        ${MY_INCLUDE_DIRECTORIES}
        ${CMAKE_CURRENT_BINARY_DIR}
       )
endif()

set(DEPENDANCY_OBJECTS "")
set(LINKED_LIBRARIES "-lpthread")
#iterate on dependancies
foreach(DEPENDANCY_ITEM IN ITEMS ${DEPENDANCY_LIST})
    message("${PROJECT_NAME} need dependancy ${DEPENDANCY_ITEM}")
    set(IS_IMPORTED FALSE)
    #Get information from dependancies if not already done
    if(NOT TARGET ${DEPENDANCY_ITEM})
        set(SUB_DEPENDANCY_OBJECTS "")
        set(SUB_LINKED_LIBRARIES "")
        add_subdirectory($ENV{QUICKY_REPOSITORY}/${DEPENDANCY_ITEM} ${CMAKE_CURRENT_BINARY_DIR}/${DEPENDANCY_ITEM})
        get_target_property(IS_IMPORTED ${DEPENDANCY_ITEM} IMPORTED)
        if(NOT ${IS_IMPORTED})
            list(APPEND DEPENDANCY_OBJECTS "$<TARGET_OBJECTS:${DEPENDANCY_ITEM}>" ${SUB_DEPENDANCY_OBJECTS})
        else()
            list(APPEND LINKED_LIBRARIES ${DEPENDANCY_ITEM})
        endif()
        list(APPEND LINKED_LIBRARIES ${SUB_LINKED_LIBRARIES})
    endif()

    # Complete include directories with directories coming from dependancy because of transitivity issues with Object libraries
    if(NOT ${IS_IMPORTED})
        set(MY_INCLUDE_DIRECTORIES ${MY_INCLUDE_DIRECTORIES} $<TARGET_PROPERTY:${DEPENDANCY_ITEM},INCLUDE_DIRECTORIES>)
    else(NOT ${IS_IMPORTED})
        set(MY_INCLUDE_DIRECTORIES ${MY_INCLUDE_DIRECTORIES} $<TARGET_PROPERTY:${DEPENDANCY_ITEM},INTERFACE_INCLUDE_DIRECTORIES>)
    endif()

endforeach(DEPENDANCY_ITEM)


#Prepare targets
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(IS_DIRECTORY ${HAS_PARENT})
    message("Declare library ${PROJECT_NAME}")
    add_library(${PROJECT_NAME} OBJECT ${MY_SOURCE_FILES} ${GENERATED_FILES})
    set(SUB_DEPENDANCY_OBJECTS ${DEPENDANCY_OBJECTS} PARENT_SCOPE)
    set(SUB_LINKED_LIBRARIES ${LINKED_LIBRARIES} PARENT_SCOPE)
else()
    #set(CMAKE_VERBOSE_MAKEFILE ON)
#    string(REPLACE " " ";" DEPENDANCY_OBJECTS ${DEPENDANCY_OBJECTS})
    add_executable(${PROJECT_NAME} ${MY_SOURCE_FILES} ${DEPENDANCY_OBJECTS} src/main.cpp)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -D$<UPPER_CASE:${PROJECT_NAME}>_SELF_TEST)
    message(Linked librarries ${LINKED_LIBRARIES})
    target_link_libraries(${PROJECT_NAME} ${LINKED_LIBRARIES})
    target_compile_options(${PROJECT_NAME} PUBLIC -Wall -ansi $<$<CONFIG:Debug>:-O0> ${MY_CPP_FLAGS})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${MY_INCLUDE_DIRECTORIES})

foreach(DEPENDANCY_ITEM IN ITEMS ${DEPENDANCY_LIST})
    add_dependencies(${PROJECT_NAME} ${DEPENDANCY_ITEM})
endforeach(DEPENDANCY_ITEM)

#Force use of -std=c++11 instead of -std=gnu++11
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)

if(DEFINED CMAKE_CUDA_HOST_COMPILER)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
        get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY COMPILE_OPTIONS)
        if(NOT "${old_flags}" STREQUAL "")
            get_property(old_cxx_standard TARGET ${EXISTING_TARGET} PROPERTY CXX_STANDARD)
            string(REPLACE ";" "," CUDA_flags "${old_flags}")
            # C++ standard set to 11 and not propagated due to issues when compiling with -std=c++17
            set_property(TARGET ${EXISTING_TARGET} PROPERTY COMPILE_OPTIONS
		                 "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}$<$<BOOL:${old_cxx_standard}>:,-std=c++11>$<$<CONFIG:Debug>:,-g>>"
                        )
        endif()
    endfunction()
    CUDA_CONVERT_FLAGS(${PROJECT_NAME})
endif(DEFINED CMAKE_CUDA_HOST_COMPILER)
#EOF
