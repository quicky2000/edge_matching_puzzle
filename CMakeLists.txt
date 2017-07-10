cmake_minimum_required(VERSION 3.8)
project(edge_matching_puzzle)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 11)

set(MY_SOURCE_FILES
    include/algo_based_feature.h
    include/border_backtracker.h
    include/border_color_constraint.h
    include/border_constraint_generator.h
    include/border_exploration.h
    include/emp_basic_strategy_generator.h
    include/emp_constraint.h
    include/emp_FSM.h
    include/emp_FSM_context.h
    include/emp_FSM_info.h
    include/emp_FSM_motor.h
    include/emp_FSM_situation.h
    include/emp_FSM_situation_analyzer.h
    include/emp_FSM_transition.h
    include/emp_gui.h
    include/emp_link.h
    include/emp_piece.h
    include/emp_piece_border.h
    include/emp_piece_constraint.h
    include/emp_piece_corner.h
    include/emp_piece_db.h
    include/emp_pieces_parser.h
    include/emp_position_strategy.h
    include/emp_situation_binary_dumper.h
    include/emp_situation_binary_reader.h
    include/emp_spiral_strategy_generator.h
    include/emp_strategy.h
    include/emp_strategy_generator.h
    include/emp_stream_strategy_generator.h
    include/emp_text_strategy_generator.h
    include/emp_types.h
    include/emp_web_server.h
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
    include/feature_simplex.h
    include/light_border_pieces_db.h
    include/octet_array.h
    include/precomputed_constraint.h
    include/precomputed_transition_info.h
    include/sequential_border_backtracker.h
    src/border_backtracker.cpp
    src/emp_FSM.cpp
    src/emp_FSM_motor.cpp
    src/emp_FSM_situation.cpp
    src/emp_FSM_situation_analyzer.cpp
    src/emp_piece.cpp
    src/emp_types.cpp
    src/emp_web_server.cpp
    src/feature_compute_stats.cpp
    src/feature_display_all.cpp
    src/feature_display_max.cpp
    src/feature_display_solutions.cpp
    src/feature_dump_solutions.cpp)


# List header directories in project
set(MY_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

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

#------------------------------
#- Generic part
#------------------------------
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
    endif()

endforeach(DEPENDANCY_ITEM)


#Prepare targets
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(IS_DIRECTORY ${HAS_PARENT})
    message("Declare library ${PROJECT_NAME}")
    add_library(${PROJECT_NAME} OBJECT ${MY_SOURCE_FILES})
    set(SUB_DEPENDANCY_OBJECTS ${DEPENDANCY_OBJECTS} PARENT_SCOPE)
    set(SUB_LINKED_LIBRARIES ${LINKED_LIBRARIES} PARENT_SCOPE)
else()
    #set(CMAKE_VERBOSE_MAKEFILE ON)
#    string(REPLACE " " ";" DEPENDANCY_OBJECTS ${DEPENDANCY_OBJECTS})
    add_executable(${PROJECT_NAME} ${MY_SOURCE_FILES} ${DEPENDANCY_OBJECTS} src/main.cpp)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -D$<UPPER_CASE:${PROJECT_NAME}>_SELF_TEST)
    message(Linked librarries ${LINKED_LIBRARIES})
    target_link_libraries(${PROJECT_NAME} ${LINKED_LIBRARIES})
    target_compile_options(${PROJECT_NAME} PUBLIC -Wall -ansi -pedantic -g -O0)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${MY_INCLUDE_DIRECTORIES})

foreach(DEPENDANCY_ITEM IN ITEMS ${DEPENDANCY_LIST})
    add_dependencies(${PROJECT_NAME} ${DEPENDANCY_ITEM})
endforeach(DEPENDANCY_ITEM)

#Force use of -std=c++11 instead of -std=gnu++11
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)

#EOF
